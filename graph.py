import json
import re
from typing import Literal

from langgraph.graph import StateGraph, END

from state import InterviewState
from model import llm


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

def generate_question_node(state: InterviewState) -> dict:
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

    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=100,
    temperature=0.7)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    updated_history = state.interview_history + [
        {"round": state.current_round, "question": question}
    ]
    return {"current_question": question, "interview_history": updated_history}


def evaluate_answer_node(state: InterviewState) -> dict:
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

    return {
        "evaluation_score": float(score) if score is not None else None,
        "evaluation_detail": details,
    }


def decide_next_step_node(state: InterviewState) -> dict:
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

    if len(recent_answers) >= 2:
        answers_block = "\n".join(f"- {a}" for a in recent_answers)
        abort_prompt = (
            "You are judging candidate engagement in a job interview.\n"
            "Here are the candidate's most recent answers:\n"
            f"{answers_block}\n\n"
            "If the candidate is clearly disengaged — e.g. giving insults, single dismissive "
            "words, random characters, or repeatedly refusing to answer — respond with "
            "exactly: ABORT\n"
            "Otherwise respond with exactly: CONTINUE"
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
            return {"interview_stage": "followup", "followup_count": state.followup_count + 1}

    return {
        "interview_stage": "question",
        "current_round": state.current_round + 1,
        "followup_count": 0,
    }


def generate_followup_node(state: InterviewState) -> dict:
    prompt = (
        "The candidate gave a weak answer. Ask a follow-up question to probe deeper.\n"
        f"Original question: {state.current_question}\n"
        f"Candidate's answer: {state.candidate_answer}\n"
        "Generate exactly one follow-up question. "
        "IMPORTANT: This is a face-to-face verbal interview. Do NOT ask the candidate "
        "to write, type, or produce any code.\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}"
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=100,
    temperature=0.7)
    followup = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    updated_history = state.interview_history + [
        {"round": state.current_round, "question": followup, "type": "followup"}
    ]
    return {"current_question": followup, "interview_history": updated_history}


def generate_report_node(state: InterviewState) -> dict:
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
        f"Role: {state.role}, Level: {state.level}\n\n"
        f"Full interview transcript:\n{history_text}\n\n"
        "The report must include:\n"
        "1. Overall assessment\n"
        "2. Key strengths demonstrated\n"
        "3. Areas for improvement\n"
        "4. Final recommendation\n"
        "Use Markdown headings and bullet points. Base the report ONLY on the transcript above.\n\n"
        f"{_lang_instruction(state.role, state.style, state.candidate_answer)}"
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=500,
    temperature=0.5)
    report = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    return {"final_report": report}


# ── Routing functions ──────────────────────────────────────────────────────────

def route_entry(state: InterviewState) -> Literal["evaluate_answer", "generate_question"]:
    """No answer → start of interview, generate first question directly."""
    if state.candidate_answer is not None:
        return "evaluate_answer"
    return "generate_question"


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
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("decide_next_step", decide_next_step_node)
workflow.add_node("generate_followup", generate_followup_node)
workflow.add_node("generate_report", generate_report_node)

workflow.set_conditional_entry_point(
    route_entry,
    {
        "evaluate_answer": "evaluate_answer",
        "generate_question": "generate_question",
    },
)

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

graph = workflow.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_chat(state: InterviewState) -> dict:
    """
    Run the interview graph from the given state.

    - Start call:  candidate_answer is None  → routes to generate_question
    - Answer call: candidate_answer is set   → routes to evaluate_answer → decide → ...

    Returns a unified response dict consumed by the backend.
    """
    final = graph.invoke(state)

    # LangGraph returns a dict for StateGraph; reconstruct the Pydantic model.
    if isinstance(final, dict):
        final_state = InterviewState(**final)
    else:
        final_state = final

    finished = final_state.interview_stage in ("finished", "aborted")
    aborted = final_state.interview_stage == "aborted"
    is_followup = final_state.interview_stage == "followup"

    return {
        "question": final_state.current_question,
        "evaluation_score": final_state.evaluation_score,
        "evaluation_detail": final_state.evaluation_detail,
        "finished": finished,
        "aborted": aborted,
        "is_followup": is_followup,
        "current_round": final_state.current_round,
        "followup_count": final_state.followup_count,
        "report": final_state.final_report if finished else None,
    }

# (run_start and run_evaluate_and_next removed — use run_chat instead)
