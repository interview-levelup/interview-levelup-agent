from langgraph import StateGraph

from state import InterviewState


def generate_question(state: InterviewState) -> InterviewState:
    # build prompt for LLM based on state
    prompt = (
        f"You are an interviewer asking a {state.level} level "
        f"question for a {state.role} role in a {state.style} style. "
        "Generate exactly one question."
    )
    # adjust difficulty with current_round
    if state.current_round > 0:
        prompt += f" Difficulty should increase with round {state.current_round}."

    # call OpenAI chat completion
    try:
        from openai import ChatCompletion
    except ImportError:
        raise

    response = ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    question = response.choices[0].message["content"].strip()

    # update state
    state.current_question = question
    state.interview_history.append({
        "round": state.current_round,
        "question": question,
    })
    return state


def evaluate_answer(state: InterviewState) -> InterviewState:
    # prepare prompt including question and candidate answer
    prompt = (
        "Evaluate the candidate's answer to the following interview question. "
        "Score between 1 and 10 based on clarity, structure, and depth. "
        "Provide a JSON object with fields 'score' and 'details'.\n"  # details can elaborate on each criterion
        f"Question: {state.current_question}\n"
        f"Answer: {state.candidate_answer}\n"
    )

    try:
        from openai import ChatCompletion
    except ImportError:
        raise

    response = ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    content = response.choices[0].message["content"].strip()

    # attempt to parse JSON from content
    import json

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # fallback: treat entire content as details with no score
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
    ...


def generate_report(state: InterviewState) -> InterviewState:
    ...


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

interview_graph.add_conditional(
    "decide_next_step",
    lambda state: state.interview_stage == "followup",
    "generate_followup",
)
interview_graph.add_conditional(
    "decide_next_step",
    lambda state: state.interview_stage == "question",
    "generate_question",
)
interview_graph.add_conditional(
    "decide_next_step",
    lambda state: state.interview_stage == "finished",
    "generate_report",
)

# connect followup and report paths
interview_graph.add_edge("generate_followup", "evaluate_answer")
interview_graph.add_edge("generate_report", None)  # terminal
