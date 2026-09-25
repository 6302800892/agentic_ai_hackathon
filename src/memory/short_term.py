"""Short-term (thread-scoped) memory: LangGraph checkpoints in SQLite (langgraph-checkpoint-sqlite).

Every super-step of a thread is checkpointed, so facts stated earlier in the same interaction (thread_id) are
available to later turns, and a thread survives process restarts.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from src.config import STATE_DIR

CHECKPOINT_DB = STATE_DIR / "checkpoints.sqlite"


def serializer():
    """Explicitly allow-list our Pydantic state types for checkpoint (de)serialisation."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from src import state

    allowed = [("src.state", name) for name, obj in vars(state).items()
               if isinstance(obj, type) and issubclass(obj, state.BaseModel) and obj.__module__ == "src.state"]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


@asynccontextmanager
async def open_checkpointer(path: Path | str = CHECKPOINT_DB):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    try:
        saver = AsyncSqliteSaver(conn, serde=serializer())
        await saver.setup()
        yield saver
    finally:
        await conn.close()
