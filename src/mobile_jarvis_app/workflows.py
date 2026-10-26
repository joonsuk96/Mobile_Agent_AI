"""Optional service guides; actions remain observation-based and bounded."""
import re
from pydantic import BaseModel, Field
from .media import displayed_views, view_requirement, meets_views, card_title

YOUTUBE_PACKAGE = "com.google.android.youtube"
CHANNEL_LINK = re.compile(r"(?:채널\s*보기|채널로\s*이동|view\s+channel|go\s+to\s+channel)", re.IGNORECASE)
VIDEO_TAB = re.compile(r"(?:동영상|videos)(?:\s*(?:탭|tab))?", re.IGNORECASE)


def youtube_content_request(command):
    """Common search/recovery applies to all requested YouTube content types."""
    return (bool(re.search(r"유튜브|유튭|youtube", command, re.IGNORECASE))
            and bool(re.search(r"영상|동영상|video|재생|틀어|음악|노래|플레이리스트|재생목록|playlist|쇼츠|shorts|라이브|\blive\b", command, re.IGNORECASE)))


def positive_mentions(command, pattern):
    """A keyword occurrence is not a request when directly negated/excluded.

    This is a routing hint, not a complete natural-language intent parser.
    The model still plans and verifies the user's full conditions.
    """
    matches = []
    for match in re.finditer(pattern, command, re.IGNORECASE):
        before = command[max(0, match.start()-24):match.start()]
        after = command[match.end():match.end()+24]
        excluded_after = re.match(r"\s*(?:(?:은|는|이|가|을|를)(?:\s*)?)?(?:말고|제외|빼고|빼줘|아닌|아니고|아니라|말구|말아|제외하고)", after)
        excluded_before = re.search(r"(?:\bno|\bnot|\bwithout|\bexcluding|\bexcept)(?:\s+(?:the|a))?\s*$", before, re.IGNORECASE)
        if not excluded_after and not excluded_before:
            matches.append(match.group())
    return matches


def youtube_format_requirement(command):
    shorts = r"쇼츠|shorts"
    longform = r"롱폼|일반\s*(?:동영상|영상)|긴\s*(?:동영상|영상)|long[ -]?form"
    if positive_mentions(command, longform):
        return {"required": "video", "excluded": ["shorts"]}
    if re.search(shorts, command, re.IGNORECASE) and not positive_mentions(command, shorts):
        return {"required": "video", "excluded": ["shorts"]}
    if positive_mentions(command, shorts):
        return {"required": "shorts", "excluded": []}
    return None


def youtube_content_kind(command):
    if positive_mentions(command, r"플레이리스트|재생목록|playlist"):
        return "playlist"
    requirement = youtube_format_requirement(command)
    if requirement:
        return requirement["required"]
    if positive_mentions(command, r"라이브|\blive\b"):
        return "live"
    if re.search(r"음악|노래", command) and not re.search(r"영상|video", command, re.IGNORECASE):
        return "music"
    return "video"


YOUTUBE_SEARCH_GUIDE = """
모든 유튜브 콘텐츠(음악, 플레이리스트, 일반 영상, 쇼츠, 라이브)의 공통 탐색 원칙:
'쇼츠 말고', '쇼츠 제외', '롱폼' 같은 형식 조건과 조회수/날짜/제목 조건을 각각 보존하고 모두 확인한다. 제외하려고 언급한 단어를 요청한 대상으로 바꾸지 않는다.
롱폼 요청이면 Shorts 선반/카드 대신 일반 영상 목록에서 후보를 확인한다. 실제 보이는 동영상 필터를 활용할 수 있다.
쇼츠 제외/롱폼을 자동으로 ‘20분 초과’ 필터로 바꾸지 않는다. 사용자가 길이를 숫자로 지정하지 않았다면 최소 분수를 임의로 추가하지 않는다.
영상의 길이·세로 썸네일만으로 Shorts 여부를 단정하지 않는다. Shorts 표시/목록 영역/일반 영상 카드와 재생 화면을 관찰하고, 불명확하면 이미지로 확인한다.
쇼츠 제외/롱폼 조건이 있으면 record_video_candidate에 content_kind와 같은 후보의 format_evidence를 제출해 형식·조회수를 따로 확인한다. unknown이거나 근거가 없으면 해당 후보를 보류하고 목록을 더 탐색한다.
형식이 다른 후보는 조회수 조건을 만족해도 선택하지 않는다. 요청한 형식을 확인하지 못했으면 성공이라고 단정하지 않는다.
검색어가 맞는 검색 결과 목록에 도착했으면 assess_youtube_screen 또는 record_youtube_search_page(target_visible=...)로 모든 조건에 맞는 목표가 보이는지 판정한다.
목표가 보이면 그 후보를 선택하고 스크롤로 지나치지 않는다. 목표가 안 보이면 스크롤/목록 스와이프를 우선한다. back/재검색/필터로 먼저 돌아가지 않는다.
검색어가 맞는 검색 결과 목록에 도착했으면 첫 화면에서 목표가 안 보인다는 이유만으로 검색 실패로 판단하지 않는다.
보이는 후보를 확인하고 부족하면 아래쪽 결과를 scroll_screen 또는 실제 목록 영역 swipe_region으로 탐색한다.
검색 입력창/키보드 화면과 결과 목록을 구분한다. 입력 화면에서는 검색 제출을 완료하고, 결과 목록에서는 추가 후보를 탐색한다.
back은 잘못된 화면/검색어, 키보드 닫기 등 구체적인 근거가 있을 때 사용한다. 같은 결과로 돌아오는 back·검색 반복을 피한다.
UI 정보가 없으면 스크린샷에서 후보와 목록 영역을 확인한다. 이미지에서 본 후보의 제목·실제 조회수는 record_youtube_search_page로 기억할 수 있다.
조회수가 '높아 보이는 것'은 화면의 실제 표시 수치를 비교하라는 상대 조건이다. 임의의 100만 같은 최소 기준을 만들거나 전체 검색 결과의 최댓값이라고 단정하지 않는다.
플레이리스트/음악 모음 요청을 무조건 특정 채널의 동영상 탭으로 바꾸지 않는다. 검색 결과의 재생목록/동영상 필터는 실제 보이고 요청에 도움이 될 때 선택한다.
영상별 조회수와 재생목록의 지표를 혼동하지 않고 실제 표시된 정보만 근거로 쓴다. 명시적인 재생목록 요청은 단일 곡 재생으로 임의 대체하지 않는다.
선호 음악은 대화에서 확인된 정보만 사용하며, 모르는 취향을 알고 있다고 주장하지 않는다.
요청한 콘텐츠가 실제 재생 중이고 현재 처리할 광고가 없으면 완료한다. 미래 광고를 기다리거나 동영상 탭으로 되돌아가지 않는다.
"""


def youtube_video_request(command):
    return (bool(re.search(r"유튜브|유튭|youtube", command, re.IGNORECASE))
            and bool(re.search(r"영상|동영상|video|재생|틀어", command, re.IGNORECASE))
            and youtube_content_kind(command) == "video")


def channel_links(screen):
    return [item for item in screen.get("elements", [])
            if CHANNEL_LINK.fullmatch(item.get("label", "").strip()) and item.get("clickable")
            and not item.get("password")]


def video_tabs(screen):
    return [item for item in screen.get("elements", [])
            if VIDEO_TAB.fullmatch(item.get("label", "").strip()) and item.get("clickable")
            and not item.get("password")]


YOUTUBE_VIDEO_GUIDE = """
유튜브 영상 탐색의 우선 실행 경로:
1. 설치된 YouTube 앱을 직접 실행한다.
2. 검색창에서 요청한 채널/인물/검색어를 입력하고 검색 결과를 연다. 기존 검색 제안을 고를 때도 요청 이름과 일치하는지 본다.
3. set_task_plan의 video_search_scope를 크리에이터/채널 검색은 channel, 일반 주제 검색은 topic으로 정한다. 일반 주제 검색을 임의의 채널 검색으로 바꾸지 않는다. 채널/인물의 올라온 영상을 찾는 요청이면 검색 결과의 '채널 보기'를 open_youtube_channel로 먼저 누른다. 버튼이 이미지에만 보이면 채널 링크를 이미지 클릭한다. 조건을 확인하기 전에 검색 결과 썸네일을 먼저 재생하지 않는다. 채널 안에서 '동영상'/'Videos' 탭을 open_youtube_videos_tab으로 연다. 이미 선택됐으면 다시 누르지 않는다.
일반 주제 검색(예: 아이언맨, 우주, 음악)은 별도 경로다. 공식 채널이 존재한다고 가정하지 않는다. 검색 결과에서 주제에 맞는 영상 카드의 실제 조회수·제목을 비교한다. 보이는 후보가 조건 미달이면 scroll_screen 또는 결과 영역 swipe_region으로 더 아래를 탐색한다. 동영상 필터가 보이면 선택할 수 있지만 채널 탭에 들어갈 필요는 없다.
같은 검색어 재입력이나 back 후 동일 결과 재진입을 반복하지 않는다. 검색어가 맞고 후보가 부족하면 우선 아직 보지 않은 결과를 탐색한다. 스크롤해도 새 결과가 없으면 실제 목록 영역/로딩/필터를 확인하고 검색어를 보완한다. '아이언맨 조회수 천만 이상' 같은 보완은 선택 사항이며 해당 문구가 조회수 필터를 적용하지는 않는다. 원래 조건은 카드의 숫자로 확인한다.
video_search_scope=auto는 채널로 확정된 상태가 아니다. 대상이 채널이 아니라는 근거가 생기면 set_youtube_search_strategy(scope="topic", reason=근거)로 즉시 전환할 수 있다. 계획을 잘못 세웠어도 목표·조건은 유지하며 경로만 수정한다.
4. 특정 채널의 최신 N번째 요청이면 해당 채널의 동영상 탭과 최신순 정렬을 확인한다. 검색 결과의 동영상 필터는 채널 동영상 탭과 다르므로, 검색 결과라면 정확한 채널을 열고 그 안의 동영상 탭을 확인한다.
5. 조회수 50만 이상 중 아무거나 같은 요청은 순번 조건이 아니다. 최신 목록을 살피되 첫 N개 순번 검증을 요구하지 않는다. 조회수는 표시 숫자로 비교하고 미달 후보는 누르지 않는다. 현재 목록에서 요청 항목/순번이 보이면 제목과 업로드 시각을 확인해 선택한다. 이미 찾은 항목을 지나치며 스크롤하지 않는다.
6. 요청 순번이 아직 화면 아래에 있거나 추가 후보가 필요할 때만 해당 영상 목록 영역을 scroll_screen/swipe_region으로 스크롤한다. 검색어/채널/정렬이 틀렸으면 스크롤로 해결하려 하지 말고 먼저 수정한다.
7. 스크롤 전후의 후보를 중복 없이 이어서 기록한다. record_ranked_candidates의 각 후보에 실제 관찰 snapshot_id와 source를 넣으면 여러 화면의 후보를 모아 검증할 수 있다. 제목이 같다고 다른 순번으로 중복 세지 않는다.
8. 요청 영상의 제목과 재생 상태를 확인하고, 광고 건너뛰기가 활성화되면 처리한다. 원래 목표가 달성되면 최종 근거를 finish_task에 모아 제출하고 즉시 끝낸다.
이미지에서 검색 결과를 확인했다면 record_youtube_search_page로 검색어와 보이는 후보의 제목·실제 조회수를 기록한다. 조건 미달 후보도 기록해 두어야 다음 화면을 탐색할 근거가 된다. UI 카드가 없다는 것은 검색 결과가 없다는 뜻이 아니다. 이미지를 보고 판단하며 같은 화면 읽기를 반복하지 않는다.
탭 이름/아이콘이 안 보이면 이미지를 확인하고 generic 도구를 사용한다. 좌표를 고정하거나 보이지 않는 탭을 추측해서 누르지 않는다.
"""


def youtube_context(command, screen, scope="auto", search_state=None):
    if not youtube_content_request(command):
        return None
    tabs = video_tabs(screen or {})
    return {
        "workflow": "youtube_content_search",
        "content_kind": youtube_content_kind(command),
        "format_requirement": youtube_format_requirement(command),
        "native_app_foreground": bool(screen and screen.get("package") == YOUTUBE_PACKAGE),
        "visible_channel_links": [item["id"] for item in channel_links(screen or {})],
        "visible_video_tabs": [{"element_id": item["id"], "selected": item.get("selected", False)} for item in tabs],
        "search_scope": scope,
        "search_memory": search_state,
        "next_step_policy": (
            "명시한 채널/크리에이터의 채널 보기 → 채널 동영상 탭. 채널이 잘못됐거나 없으면 set_youtube_search_strategy로 경로를 수정."
            if scope == "channel" and youtube_video_request(command) else
            "주제/일반 검색은 검색 결과의 영상 목록에서 찾는다. 임의 채널 진입을 강제하지 않는다. 현재 후보가 미달이면 결과 목록을 아래로 스크롤해 추가 후보를 확인. 새 후보가 없으면 목록 영역 스와이프·영상 필터·검색어 보완을 검토. 검색어와 조건을 반복 입력하거나 첫 화면으로 되돌아가는 대신 이전 탐색 기록을 활용."
        ),
    }


class SearchVideoCard(BaseModel):
    title: str = Field(min_length=1)
    view_count_text: str = Field(min_length=1, description="화면에 실제 표시된 조회수 문구")


def youtube_search_suggestion(screen, element, command):
    """Only recognize a suggestion with editor/history evidence, not a video title."""
    if screen.get("package") != YOUTUBE_PACKAGE or element.get("editable"):
        return None
    label = element.get("label", "").strip()
    if len(label) < 2 or len(label) > 100 or label not in command or re.search(r"조회수|views|채널|검색|search", label, re.IGNORECASE):
        return None
    search_ui = any(item.get("editable") for item in screen.get("elements", []))
    search_ui = search_ui or bool(re.search(r"suggest|history|search_query", element.get("resource_id", ""), re.IGNORECASE))
    search_ui = search_ui or any(re.search(r"최근\s*검색|검색\s*기록", item.get("label", "")) for item in screen.get("elements", []))
    return label if search_ui else None


class YouTubeSearchMemory:
    """Per-task observations to inform the model, never a prerecorded action path."""
    def __init__(self):
        self.query = ""
        self.searches = []
        self.candidates = {}
        self.latest_cards = []
        self.no_new_scrolls = 0
        self.last_action = None
        self.scroll_attempts = {}
        self.progress_action_id = 0

    def submit(self, query):
        self.query = query.strip()
        self.searches.append(self.query)
        self.searches = self.searches[-20:]
        self.no_new_scrolls = 0

    def observe(self, screen, action_id, action):
        if not self.query or screen.get("package") != YOUTUBE_PACKAGE:
            return
        cards = []
        added = 0
        for element in screen.get("elements", []):
            text = element.get("label", "")
            # A bare views label is metadata, not a titled candidate card.
            if not re.search(r"조회수|views", text, re.IGNORECASE):
                continue
            views = displayed_views(text)
            if not views or views["value"] is None:
                continue
            title = card_title(text)
            if not title or title.startswith("조회수") or title == views["text"]:
                continue
            key = (self.query.casefold(), title)
            added += key not in self.candidates
            item = {"title": title, "view_count": views["value"], "card_text": text}
            self.candidates[key] = item
            if len(self.candidates) > 500:
                self.candidates.pop(next(iter(self.candidates)))
            cards.append(item)
        self.latest_cards = cards
        if added:
            self.progress_action_id = action_id
        if action_id != self.last_action and (action.startswith("스크롤:") or action == "영역 스와이프"):
            key = self.query.casefold()
            self.scroll_attempts[key] = self.scroll_attempts.get(key, 0) + 1
            self.no_new_scrolls = self.no_new_scrolls + 1 if added == 0 else 0
        self.last_action = action_id

    def record_page(self, query, cards, action_id=0):
        self.query = query.strip()
        self.latest_cards = []
        discovered_new = False
        for card in cards:
            views = displayed_views(card.view_count_text)
            if not views or views["value"] is None:
                raise ValueError("화면의 실제 조회수 숫자를 적어줘.")
            item = {"title": card.title, "view_count": views["value"], "card_text": card.title + " · " + card.view_count_text}
            key = (self.query.casefold(), card.title)
            if key not in self.candidates:
                discovered_new = True
                self.progress_action_id = action_id
            self.candidates[key] = item
            self.latest_cards.append(item)
        if discovered_new:
            self.no_new_scrolls = 0

    def duplicate_query_without_exploration(self, query):
        count = sum(item.casefold().strip() == query.casefold().strip() for item in self.searches)
        return count >= 2 and self.scroll_attempts.get(query.casefold().strip(), 0) == 0

    def state(self, command):
        requirement = view_requirement(command)
        cards = [{**item, "views_match": meets_views(item["view_count"], requirement)} for item in self.latest_cards]
        query_counts = {query: self.searches.count(query) for query in dict.fromkeys(self.searches)}
        seen = [item for (query, _), item in self.candidates.items() if query == self.query.casefold()]
        if any(card["views_match"] for card in cards):
            hint = "현재 조회수 조건을 만족하는 후보가 있다. 제목의 주제/대상 조건도 맞으면 근거를 등록하고 선택. 더 검색하거나 뒤로 갈 필요가 없다."
        elif self.no_new_scrolls >= 2:
            hint = "스크롤 후 새 후보가 연속으로 없다. 전체 페이지 대신 실제 결과 목록 영역을 스와이프하거나 필터/검색어를 보완. 같은 검색어 재입력과 back 루프를 피한다."
        else:
            hint = "현재 조건에 맞는 후보가 없다면 검색 결과 아래의 후보를 더 본다. 미달 카드가 있다는 이유로 검색 실패로 판단하지 않는다. 검색어/화면이 잘못됐을 때만 back 또는 재검색."
        return {"current_query": self.query, "submitted_queries": query_counts,
                "candidates_examined_for_query": len(seen), "visible_candidates": cards,
                "scrolls_without_new_candidates": self.no_new_scrolls,
                "scroll_attempts_for_query": self.scroll_attempts.get(self.query.casefold(), 0),
                "view_requirement": requirement, "decision_hint": hint,
                "image_note": "UI 카드가 부족하면 스크린샷을 확인한다. 숫자 파서에 없다는 이유만으로 이미지의 후보가 없다고 단정하지 않는다.",
                "constraint_note": "검색어에 '조회수 천만 이상'을 넣어도 실제 필터 적용은 아니다. 실제 영상 카드 조회수를 검증해야 한다."}
