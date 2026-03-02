import json
import re
from typing import Literal

from langgraph.graph import StateGraph, END

from state import InterviewState
from model import llm


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

    score = state.evaluation_score or 0
    if score < 60 and state.followup_count < 1:
        return {"interview_stage": "followup", "followup_count": state.followup_count + 1}
    else:
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
        "to write, type, or produce any code."
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
        "Use Markdown headings and bullet points. Base the report ONLY on the transcript above."
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

    finished = final_state.interview_stage == "finished"
    is_followup = final_state.interview_stage == "followup"

    return {
        "question": final_state.current_question,
        "evaluation_score": final_state.evaluation_score,
        "evaluation_detail": final_state.evaluation_detail,
        "finished": finished,
        "is_followup": is_followup,
        "current_round": final_state.current_round,
        "followup_count": final_state.followup_count,
        "report": final_state.final_report if finished else None,
    }

# (run_start and run_evaluate_and_next removed — use run_chat instead)
