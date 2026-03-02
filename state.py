from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class InterviewState(BaseModel):
    role: str = Field(default="interviewer")
    level: str = Field(default="junior")
    style: str = Field(default="standard")
    current_round: int = Field(default=0)
    max_rounds: int = Field(default=5)
    current_question: Optional[str] = Field(default=None)
    candidate_answer: Optional[str] = Field(default=None)
    evaluation_score: Optional[float] = Field(default=None)
    evaluation_detail: Optional[Dict[str, Any]] = Field(default=None)
    followup_count: int = Field(default=0)
    interview_history: List[Any] = Field(default_factory=list)
    interview_stage: str = Field(default="question")
    final_report: Optional[str] = Field(default=None)
