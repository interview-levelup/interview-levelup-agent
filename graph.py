import json
import re
from datetime import datetime, timezone
from typing import Literal, Optional, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from logger import get_logger
from state import InterviewState
from model import llm

log = get_logger(__name__)


# ── Language detection ──────────────────────────────────────────────────────────

def _detect_script(text: str) -> str | None:
    """Return a rough script family for the dominant non-ASCII characters."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return None


def _lang_instruction(role: str, style: str = "", last_answer: str | None = None) -> str:
    """
    Returns a language rule to append to any user-facing prompt.

    Two modes:
    1. last_answer is provided (not the very first question):
       Detect the script/language of the candidate's most recent answer and
       instruct the LLM to continue in that language.  This makes the interview
       automatically mirror however the candidate is writing.
    2. No last_answer (first question):
       Let the LLM infer the most natural language from the role name alone —
       "前端工程师" → Chinese, "Frontend Engineer" → English, etc.
       No hard-coded rules: the LLM is better at reading intent than regex.
    """
    if last_answer and last_answer.strip():
        script = _detect_script(last_answer)
        if script == "chinese":
            lang_hint = "Simplified Chinese"
        elif script == "japanese":
            lang_hint = "Japanese"
        elif script == "korean":
            lang_hint = "Korean"
        else:
            # Fallback: quote the first 60 chars so the LLM can match the script itself
            sample = last_answer.strip()[:60]
            return (
                f"Language rule: The candidate's last answer was: \"{sample}\" — "
                "identify the language used and write your response in that exact language."
            )
        return (
            f"Language rule: The candidate's last answer was in {lang_hint}. "
            f"Write your response entirely in {lang_hint}."
        )
    else:
        return (
            f"Language rule: Choose the language most natural and expected for a candidate "
            f"applying for the role \"{role}\". "
            "For example, a Chinese job title implies Chinese; an English title implies English; "
            "a role explicitly tied to another language (e.g. '英语翻译', 'French interpreter') "
            "implies that target language. Do NOT default to English if the role name is in another language."
        )

# ── Node functions ─────────────────────────────────────────────────────────────
# Each node returns a *partial* dict; LangGraph merges it into the state.

def generate_question_node(state: InterviewState, config: RunnableConfig = None) -> dict:
    log.info("[generate_question] role=%s level=%s round=%d streaming=%s",
             state.role, state.level, state.current_round, config is not None)
    stream_cb: Optional[Callable[[str], None]] = (
        config.get("configurable", {}).get("stream_cb") if config else None
    )
    # Build a list of topics already covered so the LLM avoids repeating them
    previous_questions = [
        entry["question"]
        for entry in state.interview_history
        if entry.get("type") != "followup"
    ]
    avoid_block = ""
    if previous_questions:
        topics_list = "\n".join(f"- {q}" for q in previous_questions)
        avoid_block = (
            f"\n\nYou have already asked the following questions — "
            f"do NOT repeat or overlap with these topics:\n{topics_list}"
        )

    prompt = (
        f"You are an interviewer conducting a {state.style} style interview "
        f"for a {state.role} role at {state.level} level. "
        "Generate exactly one NEW question that covers a completely different topic "
        "from what has already been asked. "
        "IMPORTANT: This is a face-to-face verbal interview. Do NOT ask the candidate "
        "to write, type, or produce any code. Ask only conceptual, behavioural, or "
        "experience-based questions."
        + avoid_block
    )
    if state.current_round > 0:
        prompt += f" Difficulty should increase with round {state.current_round}."
    prompt += f"\n\n{_lang_instruction(state.role, state.style, state.candidate_answer)}"

    if stream_cb:
        question = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                stream_cb(token)
                question += token
        question = question.strip()
    else:
        response = llm.invoke(
            [{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7
        )
        question = response.content.strip() if hasattr(response, "content") else str(response).strip()

    log.debug("[generate_question] question=%r", question[:80] if question else "")
    updated_history = state.interview_history + [
        {"round": state.current_round, "question": question}
    ]
    return {"current_question": question, "interview_history": updated_history}


def evaluate_answer_node(state: InterviewState) -> dict:
    log.info("[evaluate_answer] role=%s round=%d question=%r",
             state.role, state.current_round,
             (state.current_question or "")[:60])
    prompt = (
        "Evaluate the candidate's answer to the following interview question. "
        "Score between 1 and 100 based on clarity, structure, and depth. "
        "Provide a JSON object with fields 'score' (integer 1-100) and 'details' (markdown string). "
        "The 'details' field may use markdown formatting such as bullet points and bold text.\n"
        f"Question: {state.current_question}\n"
        f"Answer: {state.candidate_answer}\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}\n"
        "Respond with ONLY the raw JSON object, no markdown code block, no extra text."
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=600,
    temperature=0.3)
    content = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Try to extract a JSON object robustly regardless of surrounding text/fences
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group(0)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        log.warning("[evaluate_answer] JSON decode failed, raw content=%r", content[:200])
        result = {"score": None, "details": content}

    score = result.get("score")
    # Some LLMs use "detail" instead of "details"
    details = result.get("details") or result.get("detail")

    # Guard: if details is itself a nested JSON string, unwrap it
    if isinstance(details, str):
        details_stripped = details.strip()
        if details_stripped.startswith("{"):
            try:
                nested = json.loads(details_stripped)
                details = nested.get("details") or nested.get("detail") or details
                if score is None and nested.get("score") is not None:
                    score = nested["score"]
            except json.JSONDecodeError:
                pass

    log.info("[evaluate_answer] score=%s", score)
    return {
        "evaluation_score": float(score) if score is not None else None,
        "evaluation_detail": details,
    }


def decide_next_step_node(state: InterviewState) -> dict:
    log.info("[decide_next_step] role=%s round=%d/%d score=%s followup_count=%d",
             state.role, state.current_round, state.max_rounds,
             state.evaluation_score, state.followup_count)
    # Finish BEFORE generating the next question when we've used all main rounds.
    # current_round is 0-indexed and only increments on main questions, so
    # finishing when current_round >= max_rounds - 1 gives exactly max_rounds total.
    if state.current_round >= state.max_rounds - 1:
        return {"interview_stage": "finished"}

    # ── Abort check: detect a disengaged candidate ────────────────────────────
    # Collect up to the last 3 answers (history + current answer)
    past_answers = [
        entry["answer"]
        for entry in state.interview_history
        if entry.get("answer")
    ]
    recent_answers = (past_answers + [state.candidate_answer or ""])[-3:]

    if len(recent_answers) >= 3:
        answers_block = "\n".join(f"- {a}" for a in recent_answers)
        abort_prompt = (
            "You are judging candidate engagement in a job interview.\n"
            "Here are the candidate's most recent answers:\n"
            f"{answers_block}\n\n"
            "Respond with ABORT only if the candidate is being actively hostile or abusive — "
            "for example: sending insults or profanity directed at the interviewer, "
            "deliberate gibberish / keyboard-mashing with clear bad intent, or explicit "
            "refusals combined with hostility (e.g. '滚', 'f*** this', '垃圾面试').\n\n"
            "Do NOT respond with ABORT for any of these — they are just weak performance:\n"
            "  - Saying they don't know ('我不会', '不知道', 'I don't know', '不清楚')\n"
            "  - Asking to skip or move on ('下一题', 'pass', '跳过', 'next question')\n"
            "  - Giving short or incomplete answers\n"
            "  - Staying silent or giving a blank response\n"
            "  - Admitting confusion or uncertainty\n\n"
            "Respond with exactly: ABORT or CONTINUE"
        )
        abort_resp = llm.invoke(
            [{"role": "user", "content": abort_prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        abort_text = (
            abort_resp.content.strip()
            if hasattr(abort_resp, "content")
            else str(abort_resp).strip()
        ).upper()
        if "ABORT" in abort_text:
            log.warning("[decide_next_step] ABORT triggered | role=%s round=%d",
                        state.role, state.current_round)
            return {"interview_stage": "aborted"}

    # ── Followup worthiness check ─────────────────────────────────────────────
    score = state.evaluation_score or 0
    if score < 60 and state.followup_count < 1:
        followup_prompt = (
            "You are a technical interviewer deciding whether a follow-up question adds value.\n"
            f"Question: {state.current_question}\n"
            f"Candidate's answer: {state.candidate_answer}\n\n"
            "A follow-up is worthwhile ONLY if the candidate gave a partially relevant answer "
            "that could be probed for more depth.\n"
            "A follow-up is NOT worthwhile if the answer is an insult, completely off-topic, "
            "nonsensical, or a single short dismissive word (e.g. '滚', 'no', '不').\n"
            "Respond with exactly: YES or NO"
        )
        followup_resp = llm.invoke(
            [{"role": "user", "content": followup_prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        followup_text = (
            followup_resp.content.strip()
            if hasattr(followup_resp, "content")
            else str(followup_resp).strip()
        ).upper()
        if "YES" in followup_text:
            log.info("[decide_next_step] -> followup")
            return {"interview_stage": "followup", "followup_count": state.followup_count + 1}

    next_round = state.current_round + 1
    log.info("[decide_next_step] -> next question (round %d)", next_round)
    return {
        "interview_stage": "question",
        "current_round": next_round,
        "followup_count": 0,
    }


def generate_followup_node(state: InterviewState, config: RunnableConfig = None) -> dict:
    log.info("[generate_followup] role=%s round=%d score=%s",
             state.role, state.current_round, state.evaluation_score)
    stream_cb: Optional[Callable[[str], None]] = (
        config.get("configurable", {}).get("stream_cb") if config else None
    )
    prompt = (
        "The candidate gave a weak answer. Ask a follow-up question to probe deeper.\n"
        f"Original question: {state.current_question}\n"
        f"Candidate's answer: {state.candidate_answer}\n"
        "Generate exactly one follow-up question. "
        "IMPORTANT: This is a face-to-face verbal interview. Do NOT ask the candidate "
        "to write, type, or produce any code.\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}"
    )
    if stream_cb:
        followup = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                stream_cb(token)
                followup += token
        followup = followup.strip()
    else:
        response = llm.invoke(
            [{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7
        )
        followup = response.content.strip() if hasattr(response, "content") else str(response).strip()

    updated_history = state.interview_history + [
        {"round": state.current_round, "question": followup, "type": "followup"}
    ]
    return {"current_question": followup, "interview_history": updated_history}


def generate_report_node(state: InterviewState, config: RunnableConfig = None) -> dict:
    log.info("[generate_report] role=%s stage=%s rounds_completed=%d",
             state.role, state.interview_stage, state.current_round)
    stream_cb: Optional[Callable[[str], None]] = (
        config.get("configurable", {}).get("stream_cb") if config else None
    )
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = []
    for entry in state.interview_history:
        label = "Follow-up" if entry.get("type") == "followup" else f"Q{entry['round'] + 1}"
        lines.append(f"**{label}:** {entry['question']}")
        if entry.get("answer"):
            lines.append(f"**Candidate:** {entry['answer']}")
        if entry.get("score") is not None:
            lines.append(f"**Score:** {entry['score']}/100")
        lines.append("")
    # Also include the final question/answer (current state)
    lines.append(f"**Final question:** {state.current_question}")
    lines.append(f"**Candidate:** {state.candidate_answer}")
    if state.evaluation_score is not None:
        lines.append(f"**Score:** {state.evaluation_score}/100")
    history_text = "\n".join(lines)
    prompt = (
        "Generate a comprehensive final interview report in Markdown format.\n"
        f"Role: {state.role}, Level: {state.level}\n"
        f"Interview date: {today}\n\n"
        f"Full interview transcript:\n{history_text}\n\n"
        "The report must include:\n"
        "1. Overall assessment\n"
        "2. Key strengths demonstrated\n"
        "3. Areas for improvement\n"
        "4. Final recommendation\n"
        "Use Markdown headings and bullet points. Base the report ONLY on the transcript above.\n\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}"
    )
    if stream_cb:
        report = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}], max_tokens=500, temperature=0.5):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                stream_cb(token)
                report += token
        report = report.strip()
    else:
        response = llm.invoke(
            [{"role": "user", "content": prompt}], max_tokens=500, temperature=0.5
        )
        report = response.content.strip() if hasattr(response, "content") else str(response).strip()

    return {"final_report": report}


# ── Routing functions ──────────────────────────────────────────────────────────

def check_sub_node(state: InterviewState) -> dict:
    """
    Let the LLM decide whether the candidate is:
    - ending the interview entirely (END)
    - directing a sub-question back at the interviewer (SUB)
    - actually attempting to answer (ANSWER)
    """
    log.info("[check_sub] role=%s round=%d answer=%r",
             state.role, state.current_round,
             (state.candidate_answer or "")[:60])
    prompt = (
        "You are an interviewer in a job interview.\n"
        f"You asked: {state.current_question}\n"
        f"The candidate replied: {state.candidate_answer}\n\n"
        "Classify the candidate's reply using these strict rules:\n\n"
        "END — the candidate wants to stop/terminate the whole interview entirely.\n"
        "  This includes explicit requests to end, quit, or stop the interview itself — "
        "not just this question.\n"
        "  e.g. '结束面试', '我想结束了', '不想继续面试了', '不想接受面试了',\n"
        "  'end the interview', 'I want to quit', 'I\'m done with this interview'.\n\n"
        "SUB — the candidate is asking the interviewer something about THIS specific question.\n"
        "  e.g. '能举个例子吗', '你的意思是…?', '这题能再说清楚点吗',\n"
        "  'could you give an example?', 'what do you mean by X?', 'can you rephrase?'\n\n"
        "ANSWER — everything else, including:\n"
        "  - Weak/partial answers or admissions of not knowing\n"
        "  - Requests to skip this one question ('下一题吧', 'pass', '跳过')\n"
        "  - Giving up on answering this question\n\n"
        "Respond with exactly one word: END, SUB, or ANSWER"
    )
    resp = llm.invoke(
        [{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    text = (resp.content.strip() if hasattr(resp, "content") else str(resp).strip()).upper()
    log.info("[check_sub] classification=%s", text)
    if "END" in text:
        return {"interview_stage": "user_end"}
    if "SUB" in text:
        return {"interview_stage": "sub"}
    return {"interview_stage": "evaluating"}


def handle_sub_node(state: InterviewState, config: RunnableConfig = None) -> dict:
    """
    Respond to the candidate's sub-question / request directed back at the interviewer.
    This could be a request for clarification, an example, more context, a rephrasing,
    or any other interviewer-directed inquiry. Respond helpfully and stay on the same topic.
    """
    log.info("[handle_sub] role=%s round=%d", state.role, state.current_round)
    stream_cb: Optional[Callable[[str], None]] = (
        config.get("configurable", {}).get("stream_cb") if config else None
    )
    prompt = (
        "You are an interviewer conducting a job interview. "
        "The candidate has responded to your question not with an answer, "
        "but with a question or request directed back at you.\n"
        f"Your original question: {state.current_question}\n"
        f"Candidate's message: {state.candidate_answer}\n\n"
        "Address the candidate's request helpfully — clarify, give an example, provide context, "
        "or rephrase as appropriate. Do NOT move to a new topic. "
        "End your response with the question restated (or refined) so the candidate knows what to answer.\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}"
    )
    if stream_cb:
        response_text = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.5):
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                stream_cb(token)
                response_text += token
        response_text = response_text.strip()
    else:
        resp = llm.invoke(
            [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.5
        )
        response_text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    return {"current_question": response_text}


def route_entry(state: InterviewState) -> str:
    """No answer → start of interview, generate first question directly."""
    if state.candidate_answer is not None:
        return "check_sub"
    return "generate_question"


def route_after_check(state: InterviewState) -> str:
    if state.interview_stage == "user_end":
        return "generate_report"
    if state.interview_stage == "sub":
        return "handle_sub"
    return "evaluate_answer"


def route_after_decide(
    state: InterviewState,
) -> Literal["generate_question", "generate_followup", "generate_report"]:
    return {
        "question": "generate_question",
        "followup": "generate_followup",
        "finished": "generate_report",
        "aborted": "generate_report",   # aborted also generates a report
    }[state.interview_stage]


# ── Build and compile the graph ────────────────────────────────────────────────

workflow = StateGraph(InterviewState)

workflow.add_node("generate_question", generate_question_node)
workflow.add_node("check_sub", check_sub_node)
workflow.add_node("handle_sub", handle_sub_node)
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("decide_next_step", decide_next_step_node)
workflow.add_node("generate_followup", generate_followup_node)
workflow.add_node("generate_report", generate_report_node)

workflow.set_conditional_entry_point(
    route_entry,
    {
        "check_sub": "check_sub",
        "generate_question": "generate_question",
    },
)

workflow.add_conditional_edges(
    "check_sub",
    route_after_check,
    {
        "handle_sub": "handle_sub",
        "evaluate_answer": "evaluate_answer",
        "generate_report": "generate_report",
    },
)

workflow.add_edge("handle_sub", END)
workflow.add_edge("evaluate_answer", "decide_next_step")
workflow.add_conditional_edges(
    "decide_next_step",
    route_after_decide,
    {
        "generate_question": "generate_question",
        "generate_followup": "generate_followup",
        "generate_report": "generate_report",
    },
)

workflow.add_edge("generate_question", END)
workflow.add_edge("generate_followup", END)
workflow.add_edge("generate_report", END)

log.info("Compiling LangGraph workflow ...")
graph = workflow.compile()
log.info("LangGraph workflow compiled and ready")


# ── Public API ─────────────────────────────────────────────────────────────────

def run_chat(state: InterviewState, stream_cb: Optional[Callable[[str], None]] = None) -> dict:
    """
    Run the interview graph from the given state.

    - Start call:  candidate_answer is None  → routes to generate_question
    - Answer call: candidate_answer is set   → routes to evaluate_answer → decide → ...

    Returns a unified response dict consumed by the backend.
    If stream_cb is provided, question-generating nodes will call it token-by-token.
    """
    cfg = {"configurable": {"stream_cb": stream_cb}} if stream_cb else None
    final = graph.invoke(state, config=cfg)

    # LangGraph returns a dict for StateGraph; reconstruct the Pydantic model.
    if isinstance(final, dict):
        final_state = InterviewState(**final)
    else:
        final_state = final

    finished = final_state.interview_stage in ("finished", "aborted", "user_end")
    aborted = final_state.interview_stage == "aborted"
    user_ended = final_state.interview_stage == "user_end"
    is_sub = final_state.interview_stage == "sub"
    is_followup = final_state.interview_stage == "followup"

    return {
        "question": final_state.current_question,
        "evaluation_score": final_state.evaluation_score,
        "evaluation_detail": final_state.evaluation_detail,
        "finished": finished,
        "aborted": aborted,
        "user_ended": user_ended,
        "is_sub": is_sub,
        "is_followup": is_followup,
        "current_round": final_state.current_round,
        "followup_count": final_state.followup_count,
        "report": final_state.final_report if finished else None,
    }

# (run_start and run_evaluate_and_next removed — use run_chat instead)
