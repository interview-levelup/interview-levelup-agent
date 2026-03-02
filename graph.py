import json
import re

from state import InterviewState
from model import llm


def generate_question(state: InterviewState) -> InterviewState:
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

    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=100,
    temperature=0.7)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    state.current_question = question
    state.interview_history.append({
        "round": state.current_round,
        "question": question,
    })
    return state


def evaluate_answer(state: InterviewState) -> InterviewState:
    prompt = (
        "Evaluate the candidate's answer to the following interview question. "
        "Score between 1 and 100 based on clarity, structure, and depth. "
        "Provide a JSON object with fields 'score' (integer 1-100) and 'details' (markdown string)."
        "The 'details' field may use markdown formatting such as bullet points and bold text.\n"
        f"Question: {state.current_question}\n"
        f"Answer: {state.candidate_answer}\n"
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
    details = result.get("details")

    state.evaluation_score = float(score) if score is not None else None
    state.evaluation_detail = details
    return state


def decide_next_step(state: InterviewState) -> InterviewState:
    # Finish BEFORE generating the next question when we've used all main rounds.
    # current_round is 0-indexed and only increments on main questions, so
    # finishing when current_round >= max_rounds - 1 gives exactly max_rounds total.
    if state.current_round >= state.max_rounds - 1:
        state.interview_stage = "finished"
        return state

    score = state.evaluation_score or 0
    if score < 60 and state.followup_count < 1:
        state.interview_stage = "followup"
        state.followup_count += 1
    else:
        state.interview_stage = "question"
        state.current_round += 1
        state.followup_count = 0

    return state


def generate_followup(state: InterviewState) -> InterviewState:
    prompt = (
        "The candidate gave a weak answer. Ask a follow-up question to probe deeper.\n"
        f"Original question: {state.current_question}\n"
        f"Candidate's answer: {state.candidate_answer}\n"
        "Generate exactly one follow-up question. "
        "IMPORTANT: This is a face-to-face verbal interview. Do NOT ask the candidate "
        "to write, type, or produce any code."
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=100,
    temperature=0.7)
    followup = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    state.current_question = followup
    state.interview_history.append({
        "round": state.current_round,
        "question": followup,
        "type": "followup",
    })
    return state


def generate_report(state: InterviewState) -> InterviewState:
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
        "Use Markdown headings and bullet points. Base the report ONLY on the transcript above."
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=500,
    temperature=0.5)
    report = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    state.final_report = report
    return state


# ── Public API ────────────────────────────────────────────────────────────────

def run_start(role: str, level: str, style: str, max_rounds: int) -> str:
    """Generate the very first interview question. Stateless."""
    state = InterviewState(role=role, level=level, style=style, max_rounds=max_rounds)
    state = generate_question(state)
    return state.current_question


def run_evaluate_and_next(state: InterviewState) -> dict:
    """Evaluate candidate's answer, decide next step, return result dict."""
    state = evaluate_answer(state)
    state = decide_next_step(state)

    if state.interview_stage == "finished":
        state = generate_report(state)
        return {
            "evaluation_score": state.evaluation_score,
            "evaluation_detail": state.evaluation_detail,
            "finished": True,
            "report": state.final_report,
        }

    if state.interview_stage == "followup":
        state = generate_followup(state)
        return {
            "evaluation_score": state.evaluation_score,
            "evaluation_detail": state.evaluation_detail,
            "finished": False,
            "next_question": state.current_question,
            "is_followup": True,
            "current_round": state.current_round,
            "followup_count": state.followup_count,
        }

    # stage == "question" → new main question
    state = generate_question(state)
    return {
        "evaluation_score": state.evaluation_score,
        "evaluation_detail": state.evaluation_detail,
        "finished": False,
        "next_question": state.current_question,
        "is_followup": False,
        "current_round": state.current_round,
        "followup_count": state.followup_count,
    }
