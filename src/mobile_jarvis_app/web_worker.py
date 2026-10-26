"""Subprocess-only terminal adapter; does not change agent files or decisions."""
import builtins
import json
import sys
import threading


def run():
    source, destination = sys.stdin, sys.stdout
    lock = threading.RLock()

    def event(kind, **data):
        with lock:
            destination.write(json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n")
            destination.flush()

    class Output:
        encoding = "utf-8"
        def __init__(self):
            self.buffer = ""
            self.question = ""
        def write(self, text):
            with lock:
                self.buffer += str(text)
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("AI:"):
                        self.question = line[3:].strip()
                    event("answer" if line.startswith("자비스 :") else "log", text=line)
            return len(text)
        def flush(self):
            with lock:
                if self.buffer.strip():
                    event("log", text=self.buffer.strip())
                    self.buffer = ""
        def isatty(self):
            return False

    output = Output()
    def receive(prompt=""):
        output.flush()
        event("question", text=output.question or prompt, prompt=prompt)
        output.question = ""
        line = source.readline()
        if not line:
            raise EOFError()
        return json.loads(line)["answer"]

    initial = json.loads(source.readline())
    sys.argv = ["click.py", "--command", initial["command"]]
    sys.stdout = sys.stderr = output
    builtins.input = receive
    try:
        from . import agent
        if initial.get("conversation_enabled"):
            from dotenv import load_dotenv
            from io import StringIO
            from .textio import read_local_text
            from .conversation import resolve_turn, ConversationTurn
            env_file = agent.PROJECT_DIR / ".env"
            if env_file.is_file():
                load_dotenv(stream=StringIO(read_local_text(env_file)), override=True)
            command = initial["command"]
            history = initial.get("history", [])
            print(f"[대화 기억] 최근 {len(history)}개 작업을 참고해 요청 해석")
            direct_first_action = not history and bool(agent.requested_app_name(command))
            model = None if direct_first_action else agent.build_model()
            for attempt in range(3):
                if direct_first_action and attempt == 0:
                    print("[빠른 시작] 독립적인 첫 앱 명령은 원문으로 전달")
                    decision = ConversationTurn(mode="action", command=command)
                else:
                    if model is None:
                        model = agent.build_model()
                    decision = resolve_turn(model, command, history)
                if decision.mode == "answer":
                    print("자비스 : " + " ".join(decision.message.split()))
                    return
                if decision.mode == "action":
                    event("resolved", text=decision.command)
                    print("[요청 해석]", decision.command)
                    sys.argv = ["click.py", "--command", decision.command]
                    break
                print("AI:", decision.message)
                answer = receive("답변: ").strip()
                if answer.casefold() in {"/cancel", "/quit", "취소", "중단"}:
                    print("자비스 : 작업을 취소했어.")
                    return
                command += "\n추가 설명: " + answer
            else:
                print("자비스 : 요청을 명확히 확인하지 못했어. 대상과 원하는 행동을 함께 알려줘.")
                return
            original_emit = agent.emit_final_answer
            def emit_with_outcome(session=None, result=None, reason=None):
                original_emit(session, result, reason)
                event("outcome", verified_success=bool(session and session.completed))
            agent.emit_final_answer = emit_with_outcome
        agent.main()
    except ValueError:
        event("error", text="대화의 요청을 해석하지 못했어. 대상과 원하는 행동을 조금 더 구체적으로 알려줘.")
        raise SystemExit(1) from None
    except Exception as exc:
        event("error", text=f"{type(exc).__name__}: 실행을 시작하지 못했어. 서버 환경을 확인해줘.")
        raise SystemExit(1) from None
    finally:
        output.flush()


if __name__ == "__main__":
    run()
