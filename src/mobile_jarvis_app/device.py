"""ADB transport and structured UI observations."""
import hashlib
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .textio import decode_text

class DeviceError(RuntimeError):
    pass

PACKAGE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+")
COMPONENT_RE = re.compile(r"([A-Za-z][\w]*(?:\.[\w]+)+)/(\.?[\w.$]+)")

def parse_screen(xml: bytes, current: dict) -> dict:
    root = ET.fromstring(xml)
    elements = []
    def visit(node, actionable=None, content_hint=None):
        if node.get("enabled") == "false":
            return
        resource_id = node.get("resource-id", "")
        if re.search(r"(?:shorts|reel)[_a-z]*(?:shelf|item|card|player|watch|recycler|container)|(?:shelf|item|card)[_a-z]*(?:shorts|reel)", resource_id, re.IGNORECASE):
            content_hint = "shorts"
        if node.get("clickable") == "true":
            actionable = node
        class_name = node.get("class", "")
        editable = (node.get("editable") == "true"
                    or class_name.endswith(("EditText", "AutoCompleteTextView")))
        password = node.get("password") == "true"
        label = ("비밀번호 입력란" if password else
                 node.get("text") or node.get("content-desc") or
                 (node.get("resource-id") or "텍스트 입력란" if editable else None))
        if label and node.get("package") != "com.android.systemui":
            target = node if editable else (actionable if actionable is not None else node)
            coords = re.findall(r"-?\d+", target.get("bounds", ""))
            if len(coords) == 4:
                x1, y1, x2, y2 = map(int, coords)
                if x2 > x1 and y2 > y1:
                    elements.append({
                        "id": str(len(elements)), "label": label[:300],
                        "resource_id": target.get("resource-id", ""),
                        "bounds": target.get("bounds"), "x": (x1+x2)//2, "y": (y1+y2)//2,
                        "clickable": target.get("clickable") == "true",
                        "checkable": target.get("checkable") == "true",
                        "password": node.get("password") == "true",
                        "class": target.get("class", ""),
                        "package": target.get("package") or current.get("package", ""),
                        "editable": editable,
                        "focused": node.get("focused") == "true",
                        "selected": node.get("selected") == "true" or target.get("selected") == "true",
                        "text": "" if password else node.get("text", ""),
                        "content_type_hint": content_hint,
                    })
        for child in node:
            visit(child, actionable, content_hint)
    visit(root)
    signature = hashlib.sha256(repr((current, elements)).encode()).hexdigest()
    rects = [list(map(int, re.findall(r"-?\d+", node.get("bounds", "")))) for node in root.iter()]
    rects = [rect for rect in rects if len(rect) == 4]
    viewport_area = max((max(0, r[2]-r[0])*max(0, r[3]-r[1]) for r in rects), default=0)
    shorts_player = False
    for node in root.iter():
        rect = list(map(int, re.findall(r"-?\d+", node.get("bounds", ""))))
        if (len(rect) == 4 and viewport_area and node.get("visible-to-user") != "false"
                and re.search(r"(?:reel|shorts)[_a-z]*(?:player|watch)", node.get("resource-id", ""), re.IGNORECASE)
                and max(0,rect[2]-rect[0])*max(0,rect[3]-rect[1]) >= viewport_area*.5):
            shorts_player = True
            break
    return {**current, "signature": signature, "elements": elements, "shorts_player_visible": shorts_player}

class AndroidDevice:
    def __init__(self, adb_path: Path, serial: str | None = None):
        self.adb_path, self.serial = Path(adb_path), serial
        self._ui_device = None

    def _run(self, args, timeout=30, use_serial=True):
        command = [str(self.adb_path)]
        if use_serial and self.serial:
            command += ["-s", self.serial]
        arguments = [str(arg) for arg in args]
        if arguments[:1] == ["shell"] and len(arguments) > 1:
            # adb shell executes through Android's POSIX shell. A Windows argv
            # list alone does not protect $ in nested Activity class names.
            arguments = ["shell", shlex.join(arguments[1:])]
        try:
            result = subprocess.run(command + arguments, capture_output=True,
                                    timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise DeviceError(f"ADB 실행 실패: {type(exc).__name__}") from exc
        if result.returncode:
            raise DeviceError((result.stderr or result.stdout).decode(
                "utf-8", errors="replace").strip()[:500])
        return result.stdout

    def text(self, *args, timeout=30):
        return self._run(args, timeout).decode("utf-8", errors="replace").strip()

    def connect(self):
        if not self.adb_path.is_file():
            raise DeviceError("ADB 경로를 확인해줘. 환경변수: JARVIS_ADB_PATH")
        output = decode_text(self._run(["devices"], use_serial=False), replace=True)
        available = [line.split()[0] for line in output.splitlines()
                     if len(line.split()) >= 2 and line.split()[1] == "device"]
        if self.serial:
            if self.serial not in available:
                raise DeviceError("지정한 폰이 연결되지 않았어.")
        elif len(available) == 1:
            self.serial = available[0]
        else:
            raise DeviceError("폰을 한 대 연결하거나 JARVIS_DEVICE_SERIAL을 지정해줘.")

    def cache_identity(self):
        try:
            serial = self.text("shell", "getprop", "ro.serialno").strip()
            if serial and serial.casefold() not in {"unknown", "null"}:
                return serial
        except DeviceError:
            pass
        serial = str(self.serial or "")
        match = re.fullmatch(r"adb-(.+)-[^-]+\._adb-tls-connect\._tcp\.?", serial)
        return match[1] if match else serial

    def launcher_components(self):
        output = self.text("shell", "cmd", "package", "query-activities", "--components",
                           "--user", "current", "-a", "android.intent.action.MAIN",
                           "-c", "android.intent.category.LAUNCHER")
        apps = {}
        for match in COMPONENT_RE.finditer(output):
            apps.setdefault(match[1], match[0])
        if not apps:
            raise DeviceError("실행 가능한 앱 목록을 읽지 못했어.")
        return apps

    def apk_paths(self, package):
        if not PACKAGE_RE.fullmatch(package):
            raise DeviceError("잘못된 패키지명")
        return [line.removeprefix("package:") for line in self.text(
            "shell", "pm", "path", "--user", "current", package).splitlines()
            if line.startswith("package:")]

    def pull(self, remote, local):
        self._run(["pull", remote, str(local)], timeout=180)

    def current_app(self):
        for command, markers in [
            (("activity", "activities"), ("mResumedActivity", "topResumedActivity")),
            (("window", "windows"), ("mCurrentFocus",)),
        ]:
            for line in self.text("shell", "dumpsys", *command).splitlines():
                if any(marker in line for marker in markers):
                    match = COMPONENT_RE.search(line)
                    if match:
                        return {"package": match[1], "activity": match[0]}
        return {"package": "", "activity": ""}

    def read_screen(self):
        before = self.current_app()
        # Reuse one UiAutomation connection for both observation and text input.
        # Starting the CLI dump in parallel with this server would disconnect it.
        try:
            xml = self._get_uiautomator().dump_hierarchy().encode("utf-8")
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(f"화면 읽기 실패 ({type(exc).__name__}).") from exc
        after = self.current_app()
        if before != after:
            raise DeviceError("관찰 중 앱이 바뀌었어. 화면을 다시 읽어줘.")
        try:
            return parse_screen(xml, after)
        except ET.ParseError as exc:
            raise DeviceError("화면 XML 오류") from exc

    def capture_screen(self):
        """Capture the physical display without starting a second UIAutomator client."""
        import base64
        import struct
        before = self.current_app()
        png = self._run(["exec-out", "screencap", "-p"])
        if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
            raise DeviceError("스크린샷을 읽지 못했어.")
        after = self.current_app()
        if before != after:
            raise DeviceError("촬영 중 앱이 바뀌었어. 다시 관찰해줘.")
        width, height = struct.unpack(">II", png[16:24])
        return {"image_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                "width": width, "height": height, **after}

    def launch(self, package, component):
        if not COMPONENT_RE.fullmatch(component) or component.split("/")[0] != package:
            raise DeviceError("검증되지 않은 실행 대상")
        output = self.text("shell", "am", "start", "-W", "--user", "current", "-a",
                           "android.intent.action.MAIN", "-c",
                           "android.intent.category.LAUNCHER", "-n", component)
        if re.search(r"(?im)^\s*(error|exception|securityexception)", output):
            raise DeviceError(output[:500])
        current = {}
        for _ in range(5):
            current = self.current_app()
            if current["package"] == package:
                break
            time.sleep(0.5)
        return {"requested_package": package,
                "foreground_matches": current.get("package") == package, **current}

    def search_web(self, query, browser_package):
        """Open a Google result URL in the requested installed browser on the phone."""
        from urllib.parse import urlencode
        if not query.strip() or len(query) > 2000:
            raise DeviceError("검색어는 1~2,000자로 입력해줘.")
        if not PACKAGE_RE.fullmatch(browser_package):
            raise DeviceError("잘못된 브라우저 패키지명")
        url = "https://www.google.com/search?" + urlencode({"q": query})
        output = self.text("shell", "am", "start", "-W", "--user", "current",
                           "-a", "android.intent.action.VIEW", "-d", url, "-p", browser_package)
        if re.search(r"(?im)^\s*(error|exception|securityexception)", output):
            raise DeviceError("선택한 앱에서 검색 페이지를 열지 못했어. 다른 브라우저를 확인해줘.")
        return {"query": query, "url": url, "browser_package": browser_package,
                "note": "검색 페이지 열기를 요청했어. 화면에서 결과와 로딩 상태를 확인해줘."}

    def tap(self, x, y):
        self.text("shell", "input", "tap", str(x), str(y))

    def navigate(self, direction):
        self.text("shell", "input", "keyevent", {"home": "3", "back": "4"}[direction])

    def scroll(self, direction):
        sizes = re.findall(r"(\d+)x(\d+)", self.text("shell", "wm", "size"))
        if not sizes:
            raise DeviceError("화면 크기를 읽지 못했어.")
        width, height = map(int, sizes[-1])
        start, end = (0.75, 0.3) if direction == "down" else (0.3, 0.75)
        self.text("shell", "input", "swipe", str(width//2), str(int(height*start)),
                  str(width//2), str(int(height*end)), "400")

    def swipe_region(self, x1, y1, x2, y2, duration=400):
        """Swipe within a chosen panel, instead of always swiping screen center."""
        sizes = re.findall(r"(\d+)x(\d+)", self.text("shell", "wm", "size"))
        if not sizes:
            raise DeviceError("화면 크기를 읽지 못했어.")
        width, height = map(int, sizes[-1])
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
            raise DeviceError("화면 범위를 벗어난 스와이프야.")
        if not 100 <= duration <= 1500 or (x1, y1) == (x2, y2):
            raise DeviceError("스와이프 거리나 시간을 확인해줘.")
        self.text("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration))

    def set_focused_text(self, text, clear=True):
        """Resolve an actual focused editor independently of XML element IDs."""
        client = self._get_uiautomator()
        try:
            focused = client(focused=True)
            if focused.count != 1:
                raise DeviceError("입력 초점이 있는 요소를 하나로 확인하지 못했어. 검색 입력칸을 먼저 눌러줘.")
            info = focused[0].info
            class_name = info.get("className", "")
            if info.get("password"):
                raise DeviceError("비밀번호는 자동 입력하지 않아.")
            if not (info.get("editable") or class_name.endswith(("EditText", "AutoCompleteTextView"))):
                raise DeviceError("초점은 있지만 텍스트 편집기임을 확인하지 못했어. 다른 입력 경로를 찾아줘.")
            bounds = info.get("bounds", {})
            values = [bounds.get(k) for k in ("left", "top", "right", "bottom")]
            if not all(isinstance(value, int) for value in values):
                raise DeviceError("초점 입력칸의 위치를 읽지 못했어.")
            element = {"editable": True, "password": False, "class": class_name,
                       "package": info.get("packageName", ""), "resource_id": info.get("resourceName", ""),
                       "bounds": f"[{values[0]},{values[1]}][{values[2]},{values[3]}]"}
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(f"입력 초점 조회 실패 ({type(exc).__name__})") from exc
        return self.set_text(element, text, clear=clear)

    def _get_uiautomator(self):
        """Start and reuse the UI RPC connection for screen reads and Unicode input."""
        if self._ui_device is None:
            try:
                import os
                import uiautomator2 as u2
                os.environ.setdefault("ADBUTILS_ADB_PATH", str(self.adb_path))
                self._ui_device = u2.connect(self.serial)
            except Exception as exc:
                self._ui_device = None
                raise DeviceError(
                    f"텍스트 입력 연결 실패 ({type(exc).__name__}). "
                    "폰 잠금과 ADB 연결을 확인해줘."
                ) from exc
        return self._ui_device

    def set_text(self, element: dict, text: str, clear: bool = True) -> dict:
        """Set Unicode directly through UIAutomator JSON-RPC, never adb input text."""
        if not element.get("editable"):
            raise DeviceError("텍스트를 입력할 수 있는 입력칸이 아니야.")
        if element.get("password"):
            raise DeviceError("비밀번호는 폰에서 직접 입력해줘.")
        if not isinstance(text, str) or len(text) > 10000:
            raise DeviceError("입력 내용은 10,000자 이하의 문자열이어야 해.")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
            raise DeviceError("지원하지 않는 제어 문자가 포함돼 있어.")
        coords = re.findall(r"-?\d+", element.get("bounds", ""))
        if len(coords) != 4:
            raise DeviceError("입력칸 위치를 확인하지 못했어.")
        expected_bounds = tuple(map(int, coords))
        selector = {}
        for field, argument in (
            ("resource_id", "resourceId"), ("class", "className"), ("package", "packageName")
        ):
            if element.get(field):
                selector[argument] = element[field]
        if not selector:
            raise DeviceError("입력칸을 구분할 정보가 부족해.")
        client = self._get_uiautomator()
        try:
            matches = client(**selector)
            selected = None
            for index in range(min(matches.count, 100)):
                candidate = matches[index]
                info = candidate.info
                bounds = info.get("bounds", {})
                actual_bounds = tuple(bounds.get(key) for key in ("left", "top", "right", "bottom"))
                if actual_bounds == expected_bounds:
                    if not info.get("enabled", True) or info.get("password", False):
                        raise DeviceError("사용할 수 없거나 보호된 입력칸이야.")
                    selected = candidate
                    break
            if selected is None:
                raise DeviceError("입력칸이 이동했거나 사라졌어. 화면을 다시 읽어줘.")
            existing = (selected.get_text(timeout=2) or "") if not clear else ""
            value = text if clear else existing + text
            accepted = selected.set_text(value, timeout=3)
            if accepted is False:
                raise DeviceError("앱이 텍스트 입력을 거부했어.")
        except DeviceError:
            raise
        except Exception as exc:
            # It might already have typed part/all of the value; never blindly retry.
            raise DeviceError(
                f"입력 처리 오류 ({type(exc).__name__}). "
                "일부 입력됐을 수 있으니 화면을 다시 확인해줘."
            ) from exc
        try:
            observed = selected.get_text(timeout=2)
            verified = observed == value
        except Exception:
            verified = None
        return {
            "characters": len(text), "mode": "replace" if clear else "append",
            "verified": verified,
            "note": ("입력칸에서 같은 문자열을 확인했어." if verified else
                     "앱이 값을 바꾸거나 화면이 전환됐을 수 있어. 다시 관찰하고 중복 입력하지 마."),
        }

