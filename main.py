from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from state import InterviewState
from graph import run_start, run_evaluate_and_next

app = FastAPI()


# ── Request / Response models ──────────────────────────────────────────────────

class StartRequest(BaseModel):
    role: str
    level: str = "junior"
    style: str = "standard"
    max_rounds: int = 5


class StartResponse(BaseModel):
    question: str


class HistoryEntry(BaseModel):
    """One Q&A turn already stored in the backend DB."""
    round: int
    question: str
    answer: Optional[str] = None
    score: Optional[float] = None
    type: Optional[str] = None   # "followup" if this was a follow-up question


class AnswerRequest(BaseModel):
    role: str
    level: str = "junior"
    style: str = "standard"
    max_rounds: int = 5
    current_round: int = 0
    followup_count: int = 0
    current_question: str
    answer: str
    interview_history: List[HistoryEntry] = []


class AnswerResponse(BaseModel):
    evaluation_score: Optional[float] = None
    evaluation_detail: Optional[str] = None
    finished: bool = False
    next_question: Optional[str] = None
    is_followup: bool = False
    current_round: int = 0
    followup_count: int = 0
    report: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    question = run_start(req.role, req.level, req.style, req.max_rounds)
    return StartResponse(question=question)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    state = InterviewState(
        role=req.role,
        level=req.level,
        style=req.style,
        max_rounds=req.max_rounds,
        current_round=req.current_round,
        followup_count=req.followup_count,
        current_question=req.current_question,
        candidate_answer=req.answer,
        interview_history=[e.model_dump(exclude_none=True) for e in req.interview_history],
    )
    result = run_evaluate_and_next(state)
    return AnswerResponse(**result)
