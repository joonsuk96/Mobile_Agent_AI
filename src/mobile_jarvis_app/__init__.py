"""Mobile Jarvis: LangChain tools controlling a connected Android phone."""


def main() -> None:
    from .agent import main as run
    run()
