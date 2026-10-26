"""Explicit text decoding for local Windows configuration and ADB output."""
from pathlib import Path


def decode_text(data: bytes, *, replace=False) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp949")
    last_error = None
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if replace:
        return data.decode("utf-8", errors="replace")
    raise ValueError("텍스트 인코딩을 읽을 수 없어. 파일을 UTF-8로 저장해줘.") from last_error


def read_local_text(path: Path) -> str:
    return decode_text(Path(path).read_bytes())
