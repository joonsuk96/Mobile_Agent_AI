"""A loopback-only web shell around the existing terminal agent."""
from collections import deque
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("web_static")
ACTIVE = {"running", "waiting", "cancelling"}


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("내용을 입력해줘.")
        return value


class TaskManager:
    def __init__(self, worker_command=None):
        self.lock = threading.RLock()
        self.task = None
        self.process = None
        self.conversation = deque(maxlen=20)
        self.worker_command = worker_command or [sys.executable, "-u", "-m", "mobile_jarvis_app.web_worker"]

    def _event(self, kind, **data):
        self.task["sequence"] += 1
        self.task["events"].append({"id": self.task["sequence"], "type": kind, **data})

    def _archive(self):
        if self.task and not self.task.get("archived"):
            self.conversation.append({key: self.task.get(key) for key in
                ("id", "command", "resolved_command", "answer", "status", "verified_success", "clarifications")})
            self.task["archived"] = True

    def clear_conversation(self):
        with self.lock:
            if self.task and (self.task["status"] in ACTIVE or self.process and self.process.poll() is None):
                raise HTTPException(409, "실행 중에는 새 대화를 시작할 수 없어.")
            self.conversation.clear()
            self.task = None
            return {"task": None}

    def snapshot(self, after=0):
        with self.lock:
            if not self.task:
                return {"task": None}
            task = {key: value for key, value in self.task.items() if key not in {"events", "sequence", "archived"}}
            task["events"] = [event for event in self.task["events"] if event["id"] > after]
            task["cursor"] = self.task["sequence"]
            return {"task": task}

    def start(self, command):
        with self.lock:
            if self.task and (self.task["status"] in ACTIVE or self.process and self.process.poll() is None):
                raise HTTPException(409, "실행 중인 작업을 먼저 마치거나 중단해줘.")
            if self.task:
                self._archive()
            self.task = {"id": uuid4().hex, "command": command, "status": "running", "question": None,
                         "answer": None, "events": deque(maxlen=2000), "sequence": 0,
                         "clarifications": [], "verified_success": None, "resolved_command": None}
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            try:
                self.process = subprocess.Popen(self.worker_command, cwd=ROOT, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.process.stdin.write(json.dumps({"command": command, "history": list(self.conversation)[-12:], "conversation_enabled": True}, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except OSError:
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                self.task["status"] = "error"
                self._event("error", text="에이전트 실행 프로세스를 시작하지 못했어.")
                self._archive()
                raise HTTPException(500, "에이전트 프로세스를 시작하지 못했어.") from None
            threading.Thread(target=self._read, args=(self.process, self.task["id"]), daemon=True).start()
            return self.snapshot()

    def _read(self, process, task_id):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("type") not in {"log", "answer", "question", "error", "outcome", "resolved"}:
                        continue
                except ValueError:
                    # Library diagnostic output is not a protocol message.
                    continue
                with self.lock:
                    if not self.task or self.task["id"] != task_id or self.task["status"] == "cancelled":
                        continue
                    kind = event["type"]
                    text = str(event.get("text", ""))[:16000]
                    if kind == "outcome":
                        value = event.get("verified_success")
                        self.task["verified_success"] = value if isinstance(value, bool) else None
                        continue
                    if kind == "resolved":
                        self.task["resolved_command"] = text
                        continue
                    if kind == "question":
                        self.task.update(status="waiting", question=text)
                    elif kind == "answer":
                        self.task["answer"] = text
                    elif kind == "error":
                        self.task["status"] = "error"
                    self._event(kind, text=text)
            code = process.wait()
            with self.lock:
                if self.task and self.task["id"] == task_id and self.task["status"] not in {"cancelled", "error"}:
                    self.task.update(status="finished" if code == 0 and self.task["answer"] else "error", question=None)
                    if self.task["status"] == "error":
                        self._event("error", text="실행이 종료됐어. 진행 로그에서 폰 연결이나 설정 안내를 확인해줘.")
                    self._event("status", text=self.task["status"])
                if self.task and self.task["id"] == task_id:
                    self._archive()
        finally:
            process.stdout.close()
            process.stdin.close()

    def reply(self, task_id, text):
        with self.lock:
            if not self.task or self.task["id"] != task_id:
                raise HTTPException(404, "작업을 찾지 못했어.")
            if self.task["status"] != "waiting":
                raise HTTPException(409, "지금은 답변을 기다리는 상태가 아니야.")
            try:
                self.process.stdin.write(json.dumps({"answer": text}, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                raise HTTPException(409, "작업이 이미 종료됐어.") from None
            self.task["clarifications"].append({"question": self.task["question"], "answer": text})
            self.task["clarifications"] = self.task["clarifications"][-8:]
            self.task.update(status="running", question=None)
            self._event("reply", text=text)
            return self.snapshot()

    def cancel(self, task_id=None):
        with self.lock:
            if task_id and (not self.task or self.task["id"] != task_id):
                raise HTTPException(404, "작업을 찾지 못했어.")
            if not self.task or self.task["status"] not in ACTIVE:
                return self.snapshot()
            self.task.update(status="cancelling", question=None)
            process = self.process
            self._event("log", text="중단 요청을 받았어. 에이전트 프로세스를 종료할게.")
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self.task.update(status="cancelled", question=None)
            self._event("status", text="cancelled")
            self._archive()
            return self.snapshot()


def create_app(manager=None):
    manager = manager or TaskManager()
    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.cancel()

    app = FastAPI(title="JARVIS · Command Center", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if request.headers.get("x-jarvis-client") != "web" or (origin and origin != str(request.base_url).rstrip("/")):
                return JSONResponse({"detail": "이 서버의 자비스 화면에서 요청해줘."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state(after: int = Query(0, ge=0), task_id: str = ""):
        with manager.lock:
            cursor = after if manager.task and manager.task["id"] == task_id else 0
            return manager.snapshot(cursor)

    @app.get("/api/conversation")
    def conversation():
        with manager.lock:
            return {"turns": list(manager.conversation)}

    @app.post("/api/conversation/reset")
    def reset_conversation():
        return manager.clear_conversation()

    @app.post("/api/tasks", status_code=202)
    def start(body: TextInput):
        return manager.start(body.text)

    @app.post("/api/tasks/{task_id}/reply")
    def reply(task_id: str, body: TextInput):
        return manager.reply(task_id, body.text)

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id: str):
        return manager.cancel(task_id)

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
    return app


app = create_app()
