import asyncio
import json
import threading
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
    aborted: bool = False                 # True when LLM terminates for hostile behaviour
    user_ended: bool = False              # True when candidate voluntarily ends the interview
    is_sub: bool = False                  # True when candidate directed a sub-question at the interviewer
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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Same as /chat but streams the interviewer's question via SSE as it is generated.
    Events:
      data: {"type": "token",  "content": "<token>"}   — one per LLM output token
      data: {"type": "done",   <all ChatResponse fields>}  — sent after graph finishes
      data: {"type": "error",  "message": "..."}  — on failure
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

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    result_holder: dict = {}

    def stream_cb(token: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, token)

    def run_in_thread() -> None:
        try:
            result = run_chat(state, stream_cb=stream_cb)
            result_holder.update(result)
        except Exception as exc:  # noqa: BLE001
            result_holder["_error"] = str(exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=run_in_thread, daemon=True).start()

    async def generator():
        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        if "_error" in result_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': result_holder['_error']})}\n\n"
        else:
            payload = {"type": "done", **result_holder}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

