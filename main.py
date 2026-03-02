from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from state import InterviewState
from graph import run_chat

app = FastAPI()


# ── Request / Response models ──────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    """One Q&A turn already stored in the backend DB."""
    round: int
    question: str
    answer: Optional[str] = None
    score: Optional[float] = None
    type: Optional[str] = None  # "followup" if this was a follow-up question


class ChatRequest(BaseModel):
    role: str
    level: str = "junior"
    style: str = "standard"
    max_rounds: int = 5
    current_round: int = 0
    followup_count: int = 0
    current_question: Optional[str] = None  # None -> start (no question yet)
    answer: Optional[str] = None            # None -> start
    interview_history: List[HistoryEntry] = []


class ChatResponse(BaseModel):
    question: Optional[str] = None        # next question (on start or after answer)
    evaluation_score: Optional[float] = None
    evaluation_detail: Optional[str] = None
    finished: bool = False
    aborted: bool = False                 # True when LLM terminates interview early
    is_followup: bool = False
    current_round: int = 0
    followup_count: int = 0
    report: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Unified interview endpoint.
    - No answer (current_question=None, answer=None) -> generates first question.
    - With answer -> evaluates answer, decides next step, returns result.
    """
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
    result = run_chat(state)
    return ChatResponse(**result)
