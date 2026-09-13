# Ola Domain Support Agent

Support bot system for Ola using CrewAI and AutoGen.
Answers customer policy queries from knowledge base, checks support tickets, and runs drafts through a review team before returning.
Runs completely offline with `MOCK_LLM` — no API keys or internet access requried.

---

## Quick Start

```bash
# setup virtual env
python -m venv venv
venv\Scripts\activate          # windows
# source venv/bin/activate     # mac/linux

# install reqs
pip install -r requirements.txt

# copy env config
cp .env.example .env

# run demo scripts (generates transcript files)
python demo_part1.py           # tasks 1-5 (dataset, rag, chunk eval)
python demo_part2.py           # tasks 6-10 (ticket tool, crew, guardrails)
python demo_part3.py           # task 13 (15 query eval)
python demo_part4.py           # tasks 14-16 (autogen review, gov, cache)

# run api server (optional)
uvicorn api.main:app --reload
```

---

## Telemetry Disabled

```
CREWAI_DISABLE_TELEMETRY=true
OTEL_SDK_DISABLED=true
```

Set in `.env.example` and forced at the top of `crew/agents.py` and `api/main.py` before crewai is imported so no external telemetry calls happen.

---

## Dataset Design Choices (Task 1)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `SEED` | `42` | fixed seed for reproducible data |
| `NUM_RECORDS` | `50` | spec asked for >=40 rows, 50 gives good margin |
| `resolution_time_hours` | `1.0–72.0 h` | reflects real ola SLAs (1h for major outage, up to 72h billing dispute) |
| `days_since_created` | `0–30` | 0 to 30 days as requested |
| `ESCALATED_PROB` | `0.20` | hits the required [10%, 30%] escalated band |
| Category weights | Billing 25%, Technical 20%, Account 20%, Defect 15%, General 20% | realistic ticket volume distribution |
| Status weights | Open 30%, In Progress 25%, Escalated 15%, Resolved 20%, Closed 10% | biased towards active/unresolved issues |

Regenerate and validate via: `python data/dataset.py`

---

## Groundedness Threshold Calibration (Task 4)

Measured cosine similarity using `all-MiniLM-L6-v2` against the `ola_sentence` index:

| Query | Type | Top-1 Similarity |
|-------|------|-----------------|
| "What is Ola's refund policy?" | In-scope | ~0.62 |
| "How are VIP customers handled?" | In-scope | ~0.59 |
| "What are the SLA levels?" | In-scope | ~0.61 |
| "When should a ticket be escalated?" | In-scope | ~0.57 |
| "How does Ola communicate in outages?" | In-scope | ~0.58 |
| "What is the weather in Mumbai?" | Out-of-scope | ~0.18 |
| "How do I cook biryani?" | Out-of-scope | ~0.12 |

In-scope scores hang around ~0.57-0.62 while random out-of-scope queries stay down at ~0.12-0.18.
Threshold chosen: **0.40** (clean midpoint between the two clusters).
Avoided using generic tutorial cutoffs like 0.7 since actual cosine embeddings for short queries rarely sit that high.

---

## Escalation Score Formula & Cutoff (Task 6)

```
escalation_score = 0.6 * float(escalated) + 0.4 * (days_since_created / 30)
```

- **0.6 on escalated flag**: agent decision carries the most weight.
- **0.4 on recency**: older open tickets need faster attention.
- Clamped between 0.0 and 1.0.

**Threshold: 0.33**
In the 50-ticket dataset, 80th percentile of ticket age is 25 days.
For an unescalated ticket at 25 days: `0.6*0 + 0.4*(25/30) = 0.333`.
Tickets at or above 0.33 get recommended for escalation review.

---

## Chunking Strategy Pick (Task 5)

See `transcripts/task5_chunking_eval.txt` for exact precision/recall numbers.

We picked the **sentence-based strategy** (`ola_sentence` collection).
Ola policies are written in full self-contained sentences. Fixed 200-char windowing frequently chops sentences in half, causing lower retrival precision and missing details. Sentence chunking preserved complete clauses and won out in both precision and recall.

---

## MOCK_LLM Design (Tasks 7, 14)

### CrewAI (`mock_llm/mock_llm.py`)
- Inherits from `crewai.llms.base_llm.BaseLLM`
- Uses **ReAct text loop** (`supports_function_calling -> False`)
- **Pitfall fix 1**: Only checks `user` messages after the initial task for `Observation:` so it doesnt trip over the prompt template string.
- **Pitfall fix 2**: Checks tool argument schema keys (`record_id` vs `query`) to dispatch tools, rather than matching on tool name strings.

### AutoGen (`review/autogen_review.py`)
- Custom `_MockAutogenClient` implementing `ChatCompletionClient`
- Cycles through deterministic responses for review and editing
- Outputs Pydantic `ReviewVerdict` json

---

## Least-Autonomy & Governance (Task 15)

`check_support_ticket_status` is given ONLY to `lookup_agent`.
`retrieval_agent` only gets `rag_lookup`, and `composer_agent` gets no tools at all. CrewAI only exposes tools registered directly to that agent, so other agents can't execute ticket lookups.
Verified via `assert_least_autonomy()` in `governance/least_autonomy.py`.

**Risk Classification: Medium**
The system processes customer support tickets and suggests refund & escalation actions affecting real users. It is not High risk (no healthcare, hiring or financial trades) but higher than Low risk (pure summarizaton) due to business action impact. Mitigations: strict input/output guardrails, pydantic schema validation, autogen review layer, and 500 token budget cap.

---

## Project Layout

```
ola-support-agent/
├── data/
│   ├── dataset.py          Task 1  — 50 ticket generator + assertions
│   └── knowledge_base.py   Task 2  — 12 policy docs
├── rag/
│   ├── chunker.py          Task 3  — fixed vs sentence chunkers
│   ├── embedder.py         Task 3  — all-MiniLM-L6-v2 embedder
│   ├── indexer.py          Task 3  — chromadb index builder
│   ├── retriever.py        Task 4  — top-k cosine search
│   └── generator.py        Task 4  — grounded answer generator
├── evaluation/
│   ├── chunking_eval.py    Task 5  — precision/recall eval
│   └── llm_judge.py        Task 13 — 15 query benchmark
├── mock_llm/
│   └── mock_llm.py         Tasks 7/14 — BaseLLM implementation
├── tools/
│   ├── rag_tool.py         Tasks 3–5  — crewai rag tool
│   └── ticket_tool.py      Task 6  — ticket status tool
├── crew/
│   ├── schemas.py          Task 9  — CrewResponse model
│   ├── guardrails.py       Task 10 — pii mask & injection check
│   ├── memory.py           Task 8  — session chat history
│   ├── agents.py           Task 7  — 3 crew agents
│   ├── tasks.py            Task 7  — 3 crew tasks
│   └── crew.py             Task 7  — crew kickoff pipeline
├── review/
│   └── autogen_review.py   Task 14 — autogen 0.4 reviewer team
├── governance/
│   ├── cache.py            Task 16 — in-memory query cache
│   ├── risk_classification.py Task 15 — risk docs + 500 token cap
│   └── least_autonomy.py   Task 15 — tool boundary checks
├── api/
│   ├── logger.py           Task 12 — json-lines logger
│   ├── endpoints.py        Task 11 — /ask and /add-document routes
│   ├── websocket.py        Task 11 — /ws/chat handler
│   └── main.py             Task 11 — fastapi app entry
├── transcripts/            Output transcripts from demo scripts
├── demo_part1.py           Runs Tasks 1–5
├── demo_part2.py           Runs Tasks 6–10
├── demo_part3.py           Runs Task 13
├── demo_part4.py           Runs Tasks 14–16
├── requirements.txt
└── .env.example
```

---

## Transcript Output Files

| File | What it covers |
|------|----------------|
| `task1_dataset_validation.txt` | Task 1 — ticket distribution and counts |
| `task4_rag_demo.txt` | Task 4 — in-scope vs out-of-scope answers |
| `task5_chunking_eval.txt` | Task 5 — fixed vs sentence precision/recall |
| `task6_ticket_tool_demo.txt` | Task 6 — ticket status & escalation formula |
| `task7_crew_demo.txt` | Task 7 — full crew execution with tool calls |
| `task8_memory_demo.txt` | Task 8 — multiturn session history |
| `task10_guardrails_demo.txt` | Task 10 — phone masking & injection blocking |
| `task13_evaluation.txt` | Task 13 — 15 query benchmark scores |
| `task14_autogen_demo.txt` | Task 14 — autogen approval & revision cases |
| `task15_governance_demo.txt` | Task 15 — least autonomy assert & budget error |
| `task16_cache_demo.txt` | Task 16 — cache hit 
