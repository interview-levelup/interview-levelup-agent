import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from state import InterviewState
from graph import graph_app

app = FastAPI()


class StartRequest(BaseModel):
    role: str
    level: str = "junior"
    style: str = "standard"
    max_rounds: int = 5


class StartResponse(BaseModel):
    thread_id: str
    question: str | None
    state: InterviewState


class AnswerRequest(BaseModel):
    thread_id: str
    answer: str


class AnswerResponse(BaseModel):
    thread_id: str
    question: str | None = None
    report: str | None = None
    finished: bool = False
    state: InterviewState


@app.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = InterviewState(
        role=req.role,
        level=req.level,
        style=req.style,
        max_rounds=req.max_rounds,
    )
    # Runs until interrupt_before=["evaluate_answer"], returning after first question
    result = graph_app.invoke(initial_state.model_dump(), config=config)
    state = InterviewState(**result)
    return StartResponse(thread_id=thread_id, question=state.current_question, state=state)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    # Inject the candidate's answer into the checkpointed state
    graph_app.update_state(config, {"candidate_answer": req.answer})
    # Resume from the interrupt point (evaluate_answer onward)
    result = graph_app.invoke(None, config=config)
    state = InterviewState(**result)
    if state.interview_stage == "finished":
        return AnswerResponse(
            thread_id=req.thread_id,
            report=state.final_report,
            finished=True,
            state=state,
        )
    return AnswerResponse(
        thread_id=req.thread_id,
        question=state.current_question,
        state=state,
    )
