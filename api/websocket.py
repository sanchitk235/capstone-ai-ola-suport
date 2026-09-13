"""
api/websocket.py - websocket chat endpoint (/ws/chat).
handles multiturn chat and catches disconnects gracefully.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from crew.crew import run_crew
from crew.guardrails import PromptInjectionError, mask_pii
from governance.risk_classification import BudgetExceededError
from api.logger import log_request

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str = "ws-default",
) -> None:
    """websocket handler for live chat session"""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            # Parse message — accept both JSON and plain text
            try:
                payload = json.loads(raw)
                query = payload.get("query", raw)
                record_id = payload.get("record_id")
            except json.JSONDecodeError:
                query = raw
                record_id = None

            t0 = time.perf_counter()
            trace_id = str(uuid.uuid4())

            try:
                crew_resp = run_crew(
                    query=query,
                    session_id=session_id,
                    record_id=record_id,
                )
                reply = {
                    "trace_id": trace_id,
                    "answer": crew_resp.answer,
                    "sources": crew_resp.sources,
                    "grounded": crew_resp.grounded,
                    "ticket_status": crew_resp.ticket_status,
                    "escalation_score": crew_resp.escalation_score,
                    "recommend_escalation": crew_resp.recommend_escalation,
                }
            except PromptInjectionError:
                reply = {
                    "trace_id": trace_id,
                    "error": "Request blocked: prompt-injection detected.",
                }
            except BudgetExceededError as exc:
                reply = {
                    "trace_id": trace_id,
                    "error": str(exc),
                }

            duration_ms = (time.perf_counter() - t0) * 1000

            log_request(
                query=mask_pii(query),
                response=reply.get("answer", reply.get("error", "")),
                session_id=session_id,
                trace_id=trace_id,
                duration_ms=duration_ms,
                grounded=reply.get("grounded", False),
                extra={"channel": "websocket"},
            )

            await websocket.send_text(json.dumps(reply))

    except WebSocketDisconnect:
        # Client disconnected — log it and let the server keep running
        print(
            json.dumps({
                "event": "websocket_disconnect",
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        )
        # Session memory is retained in _SESSION_STORE for potential reconnection
