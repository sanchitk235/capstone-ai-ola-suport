"""
crew/memory.py - session conversation memory using langchain ChatMessageHistory.
in-process dict keyed by session_id.
"""

from __future__ import annotations

from langchain_community.chat_message_histories import ChatMessageHistory

# in-memory session map
_SESSION_STORE: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """fetches or creates chat history for given session"""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def add_turn(session_id: str, human_text: str, ai_text: str) -> None:
    """appends one user and one agent message to session"""
    history = get_session_history(session_id)
    history.add_user_message(human_text)
    history.add_ai_message(ai_text)


def get_history_text(session_id: str) -> str:
    """formats chat history as plain text"""
    history = get_session_history(session_id)
    lines: list[str] = []
    for msg in history.messages:
        role = "Human" if msg.type == "human" else "AI"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    """resets a chat session"""
    _SESSION_STORE.pop(session_id, None)


def list_sessions() -> list[str]:
    """returns all tracked session ids"""
    return list(_SESSION_STORE.keys())


def session_message_count(session_id: str) -> int:
    """returns message count for session"""
    if session_id not in _SESSION_STORE:
        return 0
    return len(_SESSION_STORE[session_id].messages)
