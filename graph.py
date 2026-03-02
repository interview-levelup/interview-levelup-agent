import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import InterviewState
from model import llm


def generate_question(state: InterviewState) -> InterviewState:
    prompt = (
        f"You are an interviewer asking a {state.level} level "
        f"question for a {state.role} role in a {state.style} style. "
        "Generate exactly one question."
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
        "Score between 1 and 10 based on clarity, structure, and depth. "
        "Provide a JSON object with fields 'score' and 'details'.\n"
        f"Question: {state.current_question}\n"
        f"Answer: {state.candidate_answer}\n"
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=200,
    temperature=0.3)
    content = response.content.strip() if hasattr(response, 'content') else str(response).strip()

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
    # determine if interview should finish
    if state.current_round >= state.max_rounds:
        state.interview_stage = "finished"
        return state

    score = state.evaluation_score or 0
    if score < 6 and state.followup_count < 2:
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
        "Generate exactly one follow-up question."
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
    history_text = "\n".join(
        f"Round {entry['round']}: {entry['question']}" for entry in state.interview_history
    )
    prompt = (
        "Generate a final interview report summarizing the candidate's performance.\n"
        f"Role: {state.role}, Level: {state.level}\n"
        f"Interview history:\n{history_text}\n"
        "Include overall assessment, strengths, and areas for improvement."
    )
    response = llm.invoke([
        {"role": "user", "content": prompt},
    ],
    max_tokens=500,
    temperature=0.5)
    report = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    state.final_report = report
    return state


# build the workflow graph
interview_graph = StateGraph(InterviewState)

interview_graph.add_node("generate_question", generate_question)
interview_graph.add_node("evaluate_answer", evaluate_answer)
interview_graph.add_node("decide_next_step", decide_next_step)
interview_graph.add_node("generate_followup", generate_followup)
interview_graph.add_node("generate_report", generate_report)

# transitions
interview_graph.add_edge("generate_question", "evaluate_answer")
interview_graph.add_edge("evaluate_answer", "decide_next_step")

# conditional branching within decide_next_step
interview_graph.add_conditional_edges(
    "decide_next_step",
    lambda state: state.interview_stage,
    {
        "followup": "generate_followup",
        "question": "generate_question",
        "finished": "generate_report",
    },
)

# connect followup and report paths
interview_graph.add_edge("generate_followup", "evaluate_answer")
interview_graph.add_edge("generate_report", END)

interview_graph.set_entry_point("generate_question")

memory = MemorySaver()
graph_app = interview_graph.compile(
    checkpointer=memory,
    interrupt_before=["evaluate_answer"],
)
