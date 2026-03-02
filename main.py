from fastapi import FastAPI
from pydantic import BaseModel

from state import InterviewState
from graph import interview_graph

app = FastAPI()


class StartResponse(BaseModel):
    question: str | None
    state: InterviewState


class AnswerRequest(BaseModel):
    answer: str
    state: InterviewState


class AnswerResponse(BaseModel):
    question: str | None = None
    report: str | None = None
    state: InterviewState


@app.post("/start", response_model=StartResponse)
def start():
    state = InterviewState()
    state = interview_graph.run(state)
    return StartResponse(question=state.current_question, state=state)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    state = req.state
    state.candidate_answer = req.answer
    state = interview_graph.run(state)
    if state.interview_stage == "finished":
        return AnswerResponse(report=state.final_report, state=state)
    return AnswerResponse(question=state.current_question, state=state)
