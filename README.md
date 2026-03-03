# interview-levelup-agent

> **GitHub description:** LangGraph-powered AI interviewer agent — generates role-specific questions, evaluates answers, decides follow-ups, and streams responses via FastAPI SSE.

Python microservice that implements the AI interviewer brain. It exposes two FastAPI endpoints consumed by [interview-levelup-backend](../interview-levelup-backend) and drives the conversation through a LangGraph stateful graph.

## Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| AI orchestration | LangGraph + LangChain Core |
| LLM | OpenAI-compatible (configurable base URL) |
| State model | Pydantic v2 |

## Features

- **Role-agnostic** — works for any job role; the LLM infers domain, language, and appropriate question style from the role name
- **Adaptive difficulty** — questions get progressively harder each round
- **Answer evaluation** — scores answers 0–100 with a structured detail breakdown
- **Follow-up logic** — if a score is below threshold, the agent asks one targeted follow-up instead of moving on
- **Candidate sub-questions** — detects when the candidate redirects a question back at the interviewer and handles it gracefully
- **Abort detection** — terminates the session if hostile or off-topic behaviour is detected
- **Final report** — generates a structured debrief after all rounds complete
- **Language detection** — responds in the same language the role or candidate implies (Chinese, Japanese, English, etc.)
- **Token streaming** — `/chat/stream` emits SSE tokens as the LLM generates them

## Graph Architecture

```
             ┌─────────────┐
             │ route_entry │  (start / answer path)
             └──────┬──────┘
        ┌───────────┴──────────────┐
        ▼                          ▼
generate_question              check_sub
        │                          │
       END               ┌─────────┼──────────┐
                         ▼         ▼           ▼
                     handle_sub  (user_end) evaluate_answer
                         │           │          │
                        END          │    decide_next_step
                                     │     ┌────┼────┬──────────┐
                                     │     ▼    ▼    ▼          ▼
                                     │  (next) (fu) (finished) (aborted)
                                     │    │     │       │          │
                                     │    ▼     ▼       └────┬─────┘
                                     │ gen_q  gen_fu         ▼
                                     │   │      │       generate_report
                                     │  END    END            │
                                     └───────────────────────END
```

| Node | Responsibility |
|---|---|
| `generate_question` | Produce the next interview question (avoids repeating covered topics) |
| `check_sub` | Classify reply as: sub-question (SUB), voluntary exit (END), or answer (ANSWER) |
| `handle_sub` | Answer the candidate's sub-question and return the turn to them |
| `evaluate_answer` | Score the answer 0–100, produce evaluation detail |
| `decide_next_step` | Two-layer abort check (immediate hostile + cumulative), then route to follow-up / next question / report |
| `generate_followup` | Ask a targeted follow-up for a weak answer |
| `generate_report` | Produce the final structured debrief (triggered by: all rounds done, user exit, or abort) |

## Endpoints

### `POST /chat`

Blocking endpoint. Sends one complete LLM response.

**Request**
```json
{
  "role": "product manager",
  "level": "junior",
  "style": "standard",
  "max_rounds": 5,
  "current_round": 0,
  "current_question": null,
  "answer": null,
  "interview_history": []
}
```

Set `current_question` + `answer` to `null` to start a new session (generates first question).  
Populate both to submit an answer and receive the next step.

**Response**
```json
{
  "question": "Tell me about a time you handled competing stakeholder priorities.",
  "evaluation_score": null,
  "evaluation_detail": null,
  "finished": false,
  "is_followup": false,
  "is_sub": false,
  "current_round": 1,
  "followup_count": 0,
  "report": null
}
```

### `POST /chat/stream`

Same contract but streams the interviewer's question token-by-token via SSE.

```
data: {"type": "token",  "content": "Tell"}
data: {"type": "token",  "content": " me"}
...
data: {"type": "done",   "question": "...", "finished": false, ...}
```

## Local Setup

```bash
cp .env.example .env
# Set: LLM_API_KEY, LLM_BASE_URL (defaults to OpenAI), LLM_MODEL

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Docker

```bash
# Export required variables (or put them in your shell profile)
export LLM_API_KEY=your_key
export LLM_BASE_URL=https://api.siliconflow.cn/v1
export LLM_MODEL=deepseek-ai/DeepSeek-V3.2

# Start (default external port 8000)
docker compose up --build -d

# Start with a custom external port
AGENT_PORT=9090 docker compose up --build -d

# Logs
docker compose logs -f

# Stop
docker compose down
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | API key for the LLM provider |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AGENT_PORT` | `8000` | Host port mapped to the container (Docker only) |
