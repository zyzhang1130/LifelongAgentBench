from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import traceback
import logging
import copy
import os
import pathlib
import hashlib
import time

from .task import Task, DatasetItem
from src.typings import TaskRequest, TaskResponse, Session, ChatHistory
from src.utils import Server

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- ChatHistory compatibility shims -----------------------------------------
from copy import deepcopy
from src.typings import Role


class _CompatMessage:
    __slots__ = ("role", "content")

    def __init__(self, role, content):
        self.role = role
        self.content = content

    # allow dict-like access in case task code uses ["role"]
    def __getitem__(self, key):
        if key == "role":
            return self.role
        if key == "content":
            return self.content
        raise KeyError(key)

    # helpful when model_dump runs
    def model_dump(self):
        return {"role": self.role, "content": self.content}


class _CompatChatHistory:
    __slots__ = ("value",)

    def __init__(self, items=None):
        self.value = []
        for m in items or []:
            self.value.append(self._coerce(m))

    def _coerce(self, m):
        # Accept dicts, pydantic models, tuples, or raw strings
        if isinstance(m, _CompatMessage):
            return m
        if isinstance(m, dict):
            return _CompatMessage(m.get("role"), m.get("content"))
        if hasattr(m, "role") and hasattr(m, "content"):
            return _CompatMessage(getattr(m, "role"), getattr(m, "content"))
        if isinstance(m, (tuple, list)) and len(m) >= 2:
            return _CompatMessage(m[0], m[1])
        # last resort: treat as user text
        return _CompatMessage(Role.USER if hasattr(Role, "USER") else "user", str(m))

    def inject(self, msg):
        self.value.append(self._coerce(msg))

    def get_item_deep_copy(self, idx):
        # support negative indices like native implementation likely does
        return deepcopy(self.value[idx])

    # optional helpers some codebases call
    def get_item(self, idx):
        return self.value[idx]

    def get_last_item(self):
        return self.value[-1]

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)


def _materialize_chat_history_for_return(session_obj):
    """
    Ensure session_obj.chat_history is JSON-serializable and/or the expected pydantic type.
    """
    ch = getattr(session_obj, "chat_history", None)

    # Collect messages from either the shim or native type
    if isinstance(ch, _CompatChatHistory):
        src = ch.value
    elif hasattr(ch, "value"):
        src = ch.value
    else:
        src = []

    items = []
    for m in src:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
        # Make role JSON-friendly
        role = getattr(role, "value", role)
        items.append({"role": role, "content": content})

    # Prefer restoring the *real* pydantic type
    try:
        object.__setattr__(session_obj, "chat_history", ChatHistory(value=items))
    except Exception:
        session_obj.chat_history = ChatHistory(value=items)


# -----------------------------------------------------------------------------

# Server version identification
_SERVER_VERSION = "branch_complete_v2_2025-08-09T21:00:00"


def _file_digest(path):
    try:
        b = pathlib.Path(path).read_bytes()
        return hashlib.sha1(b).hexdigest()[:10]
    except Exception:
        return "unknown"


class TaskServer(Server):
    def __init__(self, router: APIRouter, task: Task[DatasetItem]) -> None:
        Server.__init__(self, router, task)
        self.task = task

        # Log server startup with version info
        logger.info(
            "🚀 TaskServer starting | version=%s | file=%s | sha1=%s | mtime=%s",
            _SERVER_VERSION,
            __file__,
            _file_digest(__file__),
            time.ctime(os.path.getmtime(__file__)),
        )

        self.router.post("/get_sample_index_list")(self.get_sample_index_list)
        self.router.post("/reset")(self.reset)
        self.router.post("/interact")(self.interact)
        self.router.post("/complete")(self.complete)
        self.router.post("/branch/complete")(self.branch_complete)
        self.router.post("/cleanup_check")(self.cleanup_check)
        self.router.post("/release")(self.release)
        self.router.post("/calculate_metric")(self.calculate_metric)

        # Health endpoint
        self.router.get("/__version")(
            lambda: {
                "version": _SERVER_VERSION,
                "file": __file__,
                "note": "branch_complete uses _extract_session_from_payload",
            }
        )

    def get_sample_index_list(self) -> TaskResponse.GetSampleIndexList:
        sample_index_list = self.task.get_sample_index_list()
        return TaskResponse.GetSampleIndexList(sample_index_list=sample_index_list)

    def reset(self, data: TaskRequest.Reset) -> TaskResponse.Reset:
        try:
            logger.info(f"Attempting to reset task with session: {data.session}")
            self.task.reset(data.session)
            logger.info("Task reset successful")
            return TaskResponse.Reset(session=data.session)
        except Exception as e:
            logger.error(f"Error in task reset: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Task reset failed: {str(e)}")

    def interact(self, data: TaskRequest.Interact) -> TaskResponse.Interact:
        self.task.interact(data.session)
        return TaskResponse.Interact(session=data.session)

    def complete(self, data: TaskRequest.Complete) -> TaskResponse.Complete:
        self.task.complete(data.session)
        return TaskResponse.Complete(session=data.session)

    def _extract_session_from_payload(self, data: dict) -> dict:
        """
        Accept one of:
        - {"session": {...}}
        - {"snapshot": {"session": {...}}}
        - {"snapshot": {"info": {"session": {...}}}}
        """
        if isinstance(data, dict):
            if "session" in data and isinstance(data["session"], dict):
                return data["session"]
            snap = data.get("snapshot", {})
            if isinstance(snap, dict):
                if "session" in snap and isinstance(snap["session"], dict):
                    return snap["session"]
                info = snap.get("info", {})
                if (
                    isinstance(info, dict)
                    and "session" in info
                    and isinstance(info["session"], dict)
                ):
                    return info["session"]
        raise KeyError("session not found in payload")

    def branch_complete(self, data: dict) -> dict:
        """
        Evaluate a candidate on an ephemeral database without touching live session.
        Input: {"snapshot": {...}, "candidate": "..."} OR {"session": {...}, "candidate": "..."}
        Output: {"session": {...}} with evaluation_record populated (always)
        """
        from uuid import uuid4
        from src.typings import Role, SampleStatus

        # Log payload for debugging
        logger.info("branch_complete payload keys=%s", list(data.keys()))

        branch_db = None  # Track ephemeral DB name for cleanup
        try:
            # 🔧 Robust session extraction
            session_copy = copy.deepcopy(self._extract_session_from_payload(data))
            candidate = data["candidate"]

            session_obj = Session(**session_copy)

            # --- normalize chat_history to an object that has .value (what task.py expects) ---
            def _extract_msgs(ch):
                if ch is None:
                    return []
                if isinstance(ch, dict):
                    # common shapes
                    return (
                        ch.get("value")
                        or ch.get("messages")
                        or ch.get("items")
                        or ch.get("__root__")
                        or []
                    )
                if hasattr(ch, "value"):
                    return list(ch.value)
                if hasattr(ch, "__root__"):
                    return list(getattr(ch, "__root__", []))
                if isinstance(ch, list):
                    return list(ch)
                try:
                    return list(ch)
                except Exception:
                    return []

            ch_in = getattr(session_obj, "chat_history", None)
            msgs = _extract_msgs(ch_in)

            # Force a compat wrapper that definitely has .value
            try:
                object.__setattr__(
                    session_obj, "chat_history", _CompatChatHistory(msgs)
                )
            except Exception:
                session_obj.chat_history = _CompatChatHistory(msgs)

            logger.info(
                "branch_complete compat chat_history: n=%d, typeof=%s, has_value=%s",
                len(msgs),
                type(session_obj.chat_history),
                hasattr(session_obj.chat_history, "value"),
            )
            # --- end normalization ---

            # Create isolated task that REUSES the same container
            isolated_task = self.task.create_isolated_copy()

            # Choose a unique ephemeral DB name for this branch
            parent_dataset = getattr(self.task, "_Task__dataset")
            sample_index = session_obj.sample_index
            orig_item = parent_dataset[sample_index]
            branch_db = f"{orig_item.database_name}__branch_{uuid4().hex[:8]}"

            # Swap the dataset item on the isolated task to use the branch DB
            iso_dataset = getattr(isolated_task, "_Task__dataset")
            iso_dataset[sample_index] = isolated_task._clone_item_with_db(
                orig_item, branch_db
            )

            try:
                # 🧱 Rebuild DB to the snapshot state (no candidate yet)
                isolated_task.restore_state_from_session(session_obj)

                # Now inject this candidate as the next agent turn and run one step
                msg = {"role": Role.AGENT, "content": candidate}
                session_obj.chat_history.inject(msg)  # shim guarantees this method
                isolated_task.interact(session_obj)

                # If not terminal after one step, force a judged completion on the branch
                if session_obj.sample_status.value in ("running", "initial"):
                    try:
                        # This provides an answer stub. For MD5 tasks it computes the DB hash now.
                        session_obj.task_output = (
                            isolated_task._get_default_task_output()
                        )
                    except Exception:
                        session_obj.task_output = {"answer": ""}
                    session_obj.sample_status = SampleStatus.COMPLETED

                # Always compute evaluation and cleanup; _complete() drops the branch DB
                isolated_task.complete(session_obj)

                # Convert shim → proper pydantic (or at least JSON-safe)
                _materialize_chat_history_for_return(session_obj)

                # Safe, fully JSON-encodable response
                ret = session_obj.model_dump()
                return JSONResponse(content=jsonable_encoder({"session": ret}))

            except Exception as e:
                # If something went wrong, try to cleanup the branch DB
                if branch_db:
                    try:
                        isolated_task.container.execute(
                            f"DROP DATABASE IF EXISTS `{branch_db}`;", database="mysql"
                        )
                    except Exception:
                        pass
                raise e

        except Exception as e:
            logger.error(f"Error in branch_complete: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a session with no evaluation on error
            return {"session": {}}

    def cleanup_check(self) -> dict:
        """Sanity check to verify ephemeral databases are cleaned up properly."""
        try:
            # Use the task's container to check for branch databases
            if hasattr(self.task, "container"):
                cleanup_result = self.task.container.check_branch_cleanup()
                # Parse the result to count remaining branch databases
                # MySQL returns results like: [('db__branch_abc123',), ('db__branch_def456',)]
                branch_count = 0
                if (
                    cleanup_result
                    and cleanup_result != "[]"
                    and "Error" not in cleanup_result
                ):
                    # Count non-empty results
                    if cleanup_result.strip() != "[]":
                        # Simple heuristic: count opening parentheses as database entries
                        branch_count = cleanup_result.count("('")

                status = "clean" if branch_count == 0 else f"leak_detected"

                return {
                    "cleanup_status": status,
                    "branch_databases_count": branch_count,
                    "raw_result": cleanup_result,
                }
            else:
                return {"cleanup_status": "no_container", "branch_databases_count": 0}

        except Exception as e:
            logger.error(f"Error in cleanup_check: {str(e)}")
            return {
                "cleanup_status": "error",
                "branch_databases_count": -1,
                "error": str(e),
            }

    def release(self) -> None:
        self.task.release()
        return

    def calculate_metric(
        self, data: TaskRequest.CalculateMetric
    ) -> TaskResponse.CalculateMetric:
        metric = self.task.calculate_metric(data.session_partial_list)
        return TaskResponse.CalculateMetric(metric=metric)

    def shutdown(self) -> None:
        self.release()

    @staticmethod
    def start_server(task: Task[DatasetItem], port: int, prefix: str) -> None:
        app = FastAPI()
        router = APIRouter()
        # Create an instance to access the shutdown method
        server_instance = TaskServer(router, task)
        app.include_router(router, prefix=prefix)
        # Add the shutdown event handler using lifespan events
        # https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
        app.add_event_handler("shutdown", server_instance.shutdown)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,
        )
