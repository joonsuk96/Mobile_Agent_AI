"""Local web console. Run: uv run python web.py (existing click.py is unchanged)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
# The existing click.py entry point must not shadow uvicorn's third-party click.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    import uvicorn
    import threading
    import webbrowser

    url = "http://127.0.0.1:8000"

    def open_console():
        try:
            opened = webbrowser.open(url, new=2)
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            print(f"브라우저를 자동으로 열지 못했어. 직접 접속해줘: {url}")

    class BrowserServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            # Open only after successful startup and binding to the local port.
            if self.started:
                threading.Thread(target=open_console, daemon=True).start()

    config = uvicorn.Config("mobile_jarvis_app.web_server:app", host="127.0.0.1", port=8000, workers=1)
    BrowserServer(config).run()
