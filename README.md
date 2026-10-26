# Mobile Jarvis

자연어 명령을 받아 연결된 안드로이드 폰의 앱을 탐색하고 조작하는 Python AI 에이전트입니다. LangChain과 OpenAI 모델이 화면 정보와 스크린샷을 바탕으로 행동을 결정하며, ADB와 UIAutomator로 앱 실행·클릭·스크롤·텍스트 입력을 수행합니다.

현재는 **PC에서 실행하며 ADB로 연결된 폰을 제어하는 구조**입니다. 터미널 또는 로컬 웹 채팅 화면에서 명령을 입력할 수 있습니다.

## 실행

Python 3.12 이상, uv, ADB와 연결된 안드로이드 폰이 필요합니다.

```powershell
uv sync
Copy-Item .env.example .env
```

이미 `.env`가 있으면 복사하지 않고 기존 파일을 사용합니다. `.env`에 `OPENAI_API_KEY`를 입력하고 필요하면 ADB 경로와 기기를 지정합니다.

```powershell
# 웹 화면
uv run python web.py

# 터미널
uv run python click.py
```

## 루트 파일과 폴더

| 파일·폴더 | 역할 |
| --- | --- |
| `click.py` | 터미널에서 에이전트를 실행하는 진입점 |
| `web.py` | 로컬 웹 서버를 시작하고 브라우저를 여는 진입점 |
| `src/mobile_jarvis_app/` | 에이전트와 폰 제어 코드 |
| `tests/` | 실행·탐색·완료 검증 등의 회귀 테스트 |
| `app_aliases.json` | 앱 이름과 검색 별칭 설정 |
| `pyproject.toml` | 프로젝트 정보, Python 버전 조건, 의존성·빌드 설정 |
| `uv.lock` | 재현 가능한 설치를 위한 의존성 잠금 파일 |
| `.python-version` | 사용할 Python 버전 지정 |
| `.env.example` | 비밀키 없이 공유하는 환경 설정 예시 |
| `.gitignore` | 비밀키·가상환경·캐시 등 Git 제외 규칙 |
| `.env` | 로컬 API 키와 실행 설정. Git 제외 |
| `.jarvis-cache/` | 기기별 앱 목록과 확인된 앱 이름 캐시. Git 제외 |
| `.venv/` | 로컬 Python 가상환경. Git 제외 |

## 주요 소스 파일

`src/mobile_jarvis_app/` 안의 파일입니다.

| 파일 | 역할 |
| --- | --- |
| `__init__.py` | 패키지 실행 함수를 에이전트에 연결 |
| `agent.py` | AI 모델·도구 구성, 작업 실행, 행동 제한, 오류 복구와 최종 답변 처리 |
| `device.py` | ADB·UIAutomator 연결, 화면 읽기·촬영, 앱 실행, 터치·스크롤·텍스트 입력 |
| `apps.py` | 설치 앱 정보 수집, 이름·별칭 검색과 기기별 캐시 관리 |
| `planning.py` | 작업 계획·완료 조건 관리와 관찰 근거 검증 |
| `comparison.py` | 기간 내 최대 조회수 요청의 후보·기간 경계·비교 근거 검사 |
| `exploration.py` | 앱 공통 탐색 진행과 반복·순환 감지 |
| `workflows.py` | 유튜브 탐색 지침, 화면·검색 경로 판별과 후보 기록 |
| `media.py` | 화면에 표시된 조회수 해석과 숫자 조건 비교 |
| `conversation.py` | 이전 대화를 참고한 후속 요청 해석과 대화 응답 처리 |
| `context_window.py` | 관찰 근거를 유지하면서 모델에 보내는 과거 응답·이미지 입력량 정리 |
| `textio.py` | 로컬 설정 파일과 ADB 출력의 텍스트 인코딩 처리 |
| `web_server.py` | 웹 API, 작업 시작·상태 조회·추가 답변·중단·대화 기록 관리 |
| `web_worker.py` | 별도 에이전트 프로세스의 터미널 입출력을 웹 이벤트에 연결 |
| `web_static/index.html` | 웹 채팅·상태·로그 화면 구조 |
| `web_static/style.css` | 웹 화면 디자인과 레이아웃 |
| `web_static/app.js` | 명령 전송, 상태 갱신, 추가 답변과 중단 등 웹 상호작용 |


