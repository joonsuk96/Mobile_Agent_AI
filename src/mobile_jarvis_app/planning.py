"""Explicit task contracts and observation-backed completion checks."""
import re
from copy import deepcopy
from typing import Literal
import unicodedata

from pydantic import BaseModel, Field
from .media import displayed_views, view_requirement, meets_views, card_title
from .comparison import comparison_requirement


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def comparable_title(text):
    # A trailing format tag may be hidden in the player. Preserve all other
    # title text and hashtags; do not use arbitrary prefix/fuzzy matching.
    return normalized(re.sub(r"(?:\s*#(?:shorts|쇼츠))+\s*$", "", text, flags=re.IGNORECASE))


def ordering_requirement(command):
    """Recognize common ordinal/latest wording without an app-specific rule."""
    ordinal = r"(?P<rank>[1-9][0-9]*)\s*(?:번째|번쨰|째)"
    match = re.search(ordinal + r"\s*(?:로)?\s*(?:최근|최신|오래된)", command)
    if not match:
        match = re.search(r"(?:최근|최신|오래된).{0,12}?" + ordinal, command)
    rank = int(match["rank"]) if match else None
    if rank is None and re.search(r"(?:두\s*번째|두번째).{0,5}(?:최근|최신)|(?:최근|최신).{0,8}두\s*번째", command):
        rank = 2
    if rank is None:
        return None
    return {"rank": rank, "order": "oldest" if "오래된" in command else "newest"}


class RankedCandidate(BaseModel):
    position: int = Field(ge=1, description="정렬이 확인된 목록의 순번")
    title: str = Field(min_length=1, description="실제 관찰한 항목 제목")
    publication_text: str = Field(min_length=1, description="실제 표시된 업로드/등록 시간 문구")
    snapshot_id: int | None = Field(default=None, description="이 후보를 실제 본 관찰 번호. 생략하면 정렬 목록 관찰 번호")
    source: Literal["ui", "image"] | None = Field(default=None, description="이 후보의 실제 관찰 출처")


class CompletionCheck(BaseModel):
    criterion_id: str
    evidence_text: str = Field(min_length=1)
    snapshot_id: int = 0
    source: Literal["ui", "image", "user", "app", "action"] = "ui"


class RankedListEvidence(BaseModel):
    snapshot_id: int
    order: Literal["newest", "oldest"]
    sort_active: bool
    sort_evidence: str
    candidates: list[RankedCandidate]
    source: Literal["ui", "image"] = "ui"


class SelectedItemEvidence(BaseModel):
    snapshot_id: int
    visible_title: str
    source: Literal["ui", "image"] = "ui"


class GoalTracker:
    def __init__(self, original_command="", enabled=True):
        self.enabled = enabled
        self.original_command = original_command
        self.goal = ""
        self.constraints = []
        self.steps = []
        self.criteria = {}
        self.observations = {}
        self.latest_snapshot = None
        self.ranked_selection = None
        self.candidate_selection = None
        self.selected_item_snapshot = None
        self.selected_item_matches = False
        self.selected_item_package = None
        self.revisions = []
        self.app_names = {}

    @property
    def ready(self):
        return bool(self.criteria)

    def set_plan(self, goal, constraints, success_criteria, steps):
        if self.ready:
            raise ValueError("원래 목표와 완료 기준은 이미 기록했어. revise_task_plan으로 실행 경로만 수정해줘.")
        if not goal.strip() or not success_criteria or not steps:
            raise ValueError("원래 목표, 완료 기준, 실행 단계를 빠짐없이 적어줘.")
        if any(not isinstance(x, str) or not x.strip() for x in [*constraints, *success_criteria, *steps]):
            raise ValueError("계획 항목은 비어 있지 않은 문자열이어야 해.")
        if len(success_criteria) > 16 or len(steps) > 30:
            raise ValueError("완료 기준은 16개, 계획 단계는 30개 이내로 정리해줘.")
        self.goal = goal.strip()
        self.constraints = list(constraints)
        self.steps = list(steps)
        self.criteria = {f"C{i}": {"description": value, "status": "pending", "evidence": None}
                         for i, value in enumerate(success_criteria, 1)}
        return self.state()

    def revise(self, reason, steps):
        if not self.ready or not reason.strip() or not steps or any(not x.strip() for x in steps):
            raise ValueError("기존 계획과 막힌 이유, 대체 실행 단계를 적어줘.")
        if len(steps) > 30:
            raise ValueError("실행 단계는 30개 이내로 적어줘.")
        self.steps = list(steps)
        self.revisions.append(reason)
        return self.state()

    def remember(self, snapshot_id, screen, image_seen=False):
        elements = screen.get("elements", [])
        self.observations[snapshot_id] = {
            "package": screen.get("package", ""), "image_seen": image_seen,
            "labels": [element.get("label", "") for element in elements if not element.get("password")],
            "ad_visible": any(re.fullmatch(r"광고(?:\s*[0-9].*)?|ad(?:vertisement)?|sponsored", element.get("label", "").strip(), re.IGNORECASE)
                              or re.search(r"광고주|광고\s*건너뛰기|skip\s+ads?", element.get("label", ""), re.IGNORECASE)
                              for element in elements if not element.get("password")),
            "cards": [element.get("label", "") for element in elements
                      if not element.get("password") and "조회수" in element.get("label", "")],
            "text": " ".join(str(e.get("label", "")) + " " + str(e.get("text", ""))
                             for e in elements if not e.get("password")),
        }
        self.latest_snapshot = snapshot_id
        if self.selected_item_matches and self.ranked_selection:
            observation = self.observations[snapshot_id]
            expected = normalized(self.ranked_selection["title"])
            text = normalized(observation["text"])
            other_title_visible = any(
                normalized(item["title"]) != expected and normalized(item["title"]) in text
                for item in self.ranked_selection["candidates"]
            )
            if (self.selected_item_package and observation["package"] != self.selected_item_package
                    or other_title_visible and expected not in text):
                self.invalidate_selection()
        while len(self.observations) > 30 and not comparison_requirement(self.original_command):
            del self.observations[next(iter(self.observations))]

    def register_candidate(self, snapshot_id, title, view_count_text, source):
        self.evidence(snapshot_id, source, title)
        self.evidence(snapshot_id, source, view_count_text)
        views = displayed_views(view_count_text)
        if not views or not meets_views(views["value"], view_requirement(self.original_command)):
            raise ValueError("실제 조회수 표시를 읽고 요청한 숫자 조건을 충족하는 후보를 골라줘.")
        self.candidate_selection = {"title": title, "card_text": title + " - " + view_count_text,
                                    "view_count": views["value"], "view_count_text": view_count_text,
                                    "snapshot_id": snapshot_id, "source": source}
        return self.candidate_selection

    def remember_candidate(self, label, snapshot_id):
        views = displayed_views(label)
        if not views:
            return None
        self.evidence(snapshot_id, "ui", label)
        requirement = view_requirement(self.original_command)
        if not meets_views(views["value"], requirement):
            raise ValueError(f"조회수 조건 미달: 표시 {views['text']}. 조건에 맞는 다른 영상을 찾아줘.")
        self.candidate_selection = {"title": card_title(label), "card_text": label,
                                    "view_count": views["value"], "view_count_text": views["text"],
                                    "snapshot_id": snapshot_id, "source": "ui"}
        return self.candidate_selection

    def find_candidate(self, visible_title):
        for snapshot_id, observation in reversed(list(self.observations.items())):
            for label in observation.get("cards", []):
                if normalized(visible_title) in normalized(label):
                    return self.remember_candidate(label, snapshot_id)
        return None

    def invalidate_selection(self):
        self.selected_item_matches = False
        self.selected_item_snapshot = None
        self.selected_item_package = None

    def evidence(self, snapshot_id, source, text):
        observation = self.observations.get(snapshot_id)
        if not observation or not normalized(text):
            raise ValueError("실제 관찰 번호와 그 화면에서 읽은 근거 문구를 적어줘.")
        if source == "ui":
            if normalized(text) not in normalized(observation["text"]):
                # Accept equivalent view-count notation, not arbitrary paraphrases.
                numeric_text = re.fullmatch(r"\s*(?:조회수\s*)?[0-9][0-9,.]*\s*(?:억|만|천|[KMB])?\s*(?:회|views)?\s*", text, re.IGNORECASE)
                asked = displayed_views(text) if numeric_text else None
                same_number = bool(asked and any(displayed_views(label) and displayed_views(label)["value"] == asked["value"]
                                                for label in observation.get("cards", [])))
                if not same_number:
                    raise ValueError(f"관찰 {snapshot_id}의 UI에 그 근거 문구가 없어. observation_index의 실제 문구/영상 카드 또는 이미지 출처를 사용해줘.")
        elif source == "app":
            if normalized(text) != normalized(observation["package"]):
                raise ValueError("관찰한 전면 앱 패키지와 근거가 일치하지 않아.")
        elif source == "image":
            if not observation["image_seen"]:
                # Correct a source label using the SAME already-read frame.
                # This does not create image evidence or perform any new I/O.
                try:
                    resolved = self.evidence(snapshot_id, "ui", text)
                except ValueError:
                    raise ValueError("해당 관찰에 이미지도, 일치하는 UI 문구도 없어. 확인하지 않은 근거로 완료할 수 없어.") from None
                return {**resolved, "requested_source": "image", "resolution": "same_frame_ui"}
        else:
            raise ValueError("관찰 근거 출처는 ui, image, app 중 하나여야 해.")
        return {"snapshot_id": snapshot_id, "source": source, "text": text}

    def resolve_record_evidence(self, criterion_id, text, snapshot_id, source, action_history):
        """Resolve a mislabeled reference using retained real witnesses, not new claims."""
        try:
            return self.evidence(snapshot_id, source, text)
        except ValueError as original_error:
            requested = self.observations.get(snapshot_id, {})
            # Resolve a display-name claim only through a successfully launched,
            # observed package. A wrong package or an unrelated UI fact stays invalid.
            description = self.criteria[criterion_id]["description"]
            package = requested.get("package", "")
            app_only = bool(re.search(r"앱|application|app|실행|열림|전면", description, re.IGNORECASE))
            suffix = r"(?:앱|application|app)?(?:이|가|을|를)?(?:전면에서|전면으로|전면에|전면)?(?:실행됨|실행중임|실행중입니다|실행중|실행했어|실행했습니다|실행|열림|열었어|열었습니다|열려있음|열려있다|활성화됨|떠있음|있음|있습니다|opened|running|launched)?(?:(?:을|를)?확인(?:함|됨|했어|했습니다)?)?"
            claim = normalized(text)
            if source == "app" and app_only and package and package in self.app_names:
                mentioned_packages = re.findall(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){2,}", text)
                if mentioned_packages and all(item == package for item in mentioned_packages):
                    return {"source": "app", "snapshot_id": snapshot_id, "text": package,
                            "resolution": "literal_observed_package"}
            if source in {"app", "ui"} and app_only and package:
                for name in self.app_names.get(package, ()):
                    identity = normalized(name)
                    if identity and claim.startswith(identity) and re.fullmatch(suffix, claim[len(identity):], re.IGNORECASE):
                        return {"source": "app", "snapshot_id": snapshot_id, "text": package,
                                "resolution": "observed_app_identity"}
            if source == "app":
                raise original_error
            for actual_id, observation in reversed(list(self.observations.items())):
                if (observation.get("package") == requested.get("package") and normalized(text)
                        and normalized(text) in normalized(observation.get("text", ""))):
                    resolved = self.evidence(actual_id, "ui", text)
                    return {**resolved, "requested_snapshot_id": snapshot_id, "resolution": "retained_ui"}
            description = self.criteria[criterion_id]["description"]
            candidate = self.candidate_selection
            if candidate:
                kind = candidate.get("content_kind")
                aliases = {"shorts": {"shorts", "쇼츠", "쇼츠형식", "shorts형식"},
                           "video": {"video", "일반영상", "일반동영상", "롱폼"}}
                format_only = (re.search(r"형식|쇼츠|shorts|롱폼|일반\s*(?:동영상|영상)", description, re.IGNORECASE)
                               and not re.search(r"재생|playback|playing", description, re.IGNORECASE))
                if (format_only and candidate.get("format_evidence") and kind in aliases
                        and (normalized(text) in aliases[kind] or normalized(text) == normalized(candidate["format_evidence"]))):
                    return {"source": "video_candidate", "snapshot_id": candidate["snapshot_id"],
                            "text": candidate["format_evidence"], "original_source": candidate["source"],
                            "resolution": "registered_format"}
                literal_views = displayed_views(text)
                same_literal_views = (bool(re.fullmatch(r"\s*(?:조회수\s*)?[0-9][0-9,.]*\s*(?:억|만|천|[KMB])?\s*(?:회|views)?\s*", text, re.IGNORECASE))
                                      and literal_views and literal_views["value"] == candidate["view_count"])
                same_requirement = ("조회수" in description and view_requirement(text) is not None
                                    and view_requirement(text) == view_requirement(self.original_command)
                                    and meets_views(candidate["view_count"], view_requirement(self.original_command)))
                if normalized(text) == normalized(candidate["title"]) or same_literal_views or same_requirement:
                    return {"source": "video_candidate", "snapshot_id": candidate["snapshot_id"],
                            "text": candidate["title"] if normalized(text) == normalized(candidate["title"]) else candidate["view_count_text"],
                            "original_source": candidate["source"], "resolution": "registered_candidate"}
            if "광고" in description and any(item.get("status") == "executed" and item.get("action") == "광고 건너뛰기" for item in action_history):
                return {"source": "action", "text": "광고 건너뛰기", "resolution": "executed_ad_skip"}
            latest = self.observations.get(self.latest_snapshot, {})
            if "재생" in description and self.selected_item_matches:
                active = re.search(r"일시\s*(?:정지|중지)|\bpause\b", latest.get("text", ""), re.IGNORECASE)
                ad = re.search(r"광고\s*(?:재생|주|남은|[0-9])|sponsored", latest.get("text", ""), re.IGNORECASE)
                if active and not ad and not latest.get("ad_visible"):
                    return {"source": "ui", "snapshot_id": self.latest_snapshot, "text": active[0], "resolution": "current_playback_control"}
            # Vision is an existing evidence source. Correcting its label does not
            # create an image; the referenced frame must actually have been sent.
            if source == "ui" and requested.get("image_seen"):
                return {"source": "image", "snapshot_id": snapshot_id, "text": text,
                        "requested_source": source, "resolution": "image_backed_observation"}
            raise original_error

    def record(self, criterion_id, status, evidence_text, snapshot_id, source, user_answers, action_history=()):
        if criterion_id not in self.criteria:
            raise ValueError("현재 계획에 없는 완료 기준 번호야.")
        item = self.criteria[criterion_id]
        if status != "satisfied":
            item.update(status=status, evidence=None)
            return self.state()
        if source == "user":
            if not evidence_text.strip() or not any(normalized(evidence_text) in normalized(value) for value in user_answers.values()):
                raise ValueError("실제 받은 사용자 답변에 없는 근거야.")
            evidence = {"source": "user", "text": evidence_text}
        elif source == "action":
            matches = [action for action in action_history if action.get("status") == "executed"
                       and normalized(action.get("action", "")) == normalized(evidence_text)]
            if not matches:
                raise ValueError("성공 실행 기록에 없는 행동이야. 시도나 추측을 실행 근거로 등록하지 마.")
            evidence = {"source": "action", "text": evidence_text}
        else:
            evidence = self.resolve_record_evidence(criterion_id, evidence_text, snapshot_id, source, action_history)
        item.update(status="satisfied", evidence=evidence)
        return self.state()

    def record_ranked(self, snapshot_id, sort_evidence, candidates, source, order=None, sort_active=True):
        requirement = ordering_requirement(self.original_command)
        if not requirement:
            raise ValueError("원래 명령에서 순번과 시간 정렬 조건을 찾지 못했어.")
        if not sort_active or (order is not None and order != requirement["order"]):
            raise ValueError("원래 요청에 맞는 정렬이 실제 활성화됐는지 먼저 확인해줘.")
        markers = (("최신", "최근", "newest", "latest", "newtoold") if requirement["order"] == "newest"
                   else ("오래된", "오래된순", "oldest", "oldtonew"))
        if not any(normalized(marker) in normalized(sort_evidence) for marker in markers):
            raise ValueError("인기/추천 순서를 시간 순서로 가정하지 마. 요청한 시간 정렬을 확인해줘.")
        self.evidence(snapshot_id, source, sort_evidence)
        by_position = {item.position: item for item in candidates}
        rank = requirement["rank"]
        if len(by_position) != len(candidates) or not all(i in by_position for i in range(1, rank + 1)):
            raise ValueError("첫 번째부터 요청 순번까지의 실제 항목을 순서대로 확인해줘.")
        titles = [normalized(by_position[i].title) for i in range(1, rank + 1)]
        if len(set(titles)) != rank:
            raise ValueError("중복 제목을 여러 순번으로 기록하지 마.")
        for i in range(1, rank + 1):
            item = by_position[i]
            candidate_snapshot = item.snapshot_id if item.snapshot_id is not None else snapshot_id
            candidate_source = item.source or source
            self.evidence(candidate_snapshot, candidate_source, item.title)
            self.evidence(candidate_snapshot, candidate_source, item.publication_text)
        self.ranked_selection = {**requirement, "sort_evidence": sort_evidence,
                                 "title": by_position[rank].title,
                                 "candidates": [by_position[i].model_dump() for i in range(1, rank + 1)]}
        self.selected_item_matches = False
        self.selected_item_snapshot = None
        return self.ranked_selection

    def observed_title(self, snapshot_id, expected_title):
        observation = self.observations.get(snapshot_id, {})
        if normalized(expected_title) and normalized(expected_title) in normalized(observation.get("text", "")):
            return expected_title
        expected = comparable_title(expected_title)
        for label in observation.get("labels", []):
            literal = card_title(label)
            if expected and comparable_title(literal) == expected:
                return literal
        return None

    def confirm_selected(self, snapshot_id, visible_title, source):
        requested_snapshot = snapshot_id
        latest = self.observations.get(self.latest_snapshot, {})
        if snapshot_id != self.latest_snapshot:
            old = self.observations.get(snapshot_id, {})
            if old.get("package") and old.get("package") == latest.get("package") and self.observed_title(self.latest_snapshot, visible_title):
                snapshot_id = self.latest_snapshot
                source = "ui"
            else:
                expected = (self.ranked_selection or self.candidate_selection or {}).get("title", "")
                if (self.selected_item_matches and normalized(visible_title) == normalized(expected)
                        and self.selected_item_package == latest.get("package")
                        and not latest.get("text", "").strip() and latest.get("image_seen")):
                    return {"matches": True, "expected_title": expected, "visible_title": visible_title,
                            "snapshot_id": self.selected_item_snapshot, "requested_snapshot_id": requested_snapshot,
                            "verification": "동일 항목의 기존 제목 대조를 유지. 다른 항목 조작 시 무효화됨."}
                self.invalidate_selection()
                raise ValueError("현재 화면의 제목이나 유지된 선택 근거로 해당 영상을 확인할 수 없어.")
        literal_title = self.observed_title(snapshot_id, visible_title)
        self.evidence(snapshot_id, "ui" if literal_title else source, literal_title or visible_title)
        requirement = ordering_requirement(self.original_command)
        if requirement:
            if not self.ranked_selection:
                raise ValueError("정렬된 후보 목록과 요청 순번을 먼저 확인해줘.")
            expected = self.ranked_selection["title"]
        else:
            candidate = self.candidate_selection
            if not candidate or (comparable_title(visible_title) != comparable_title(candidate["title"]) and normalized(visible_title) not in normalized(candidate["card_text"])):
                candidate = self.find_candidate(visible_title)
            if view_requirement(self.original_command) and not candidate:
                raise ValueError("이 영상의 조회수 조건을 검증할 이전 영상 카드가 없어. 실제 관찰한 조회수 근거를 확인해줘.")
            expected = candidate["title"] if candidate else visible_title
        self.selected_item_matches = comparable_title(visible_title) == comparable_title(expected)
        self.selected_item_snapshot = snapshot_id
        self.selected_item_package = self.observations[snapshot_id]["package"]
        return {"matches": self.selected_item_matches, "expected_title": expected,
                "visible_title": visible_title, "snapshot_id": snapshot_id,
                "requested_snapshot_id": requested_snapshot, "view_count": (self.candidate_selection or {}).get("view_count"),
                "next_step": "일치하면 나머지 완료 기준을 확인해." if self.selected_item_matches else "잘못 선택했어. 뒤로 이동하고 요청 조건에 맞는 항목을 다시 선택해."}

    def apply_completion(self, checks, ranked_list, selected_item, user_answers, action_history):
        """Commit progress atomically while retaining independently verified facts."""
        before = deepcopy(self.criteria)
        try:
            if ranked_list is not None and ordering_requirement(self.original_command):
                self.record_ranked(ranked_list.snapshot_id, ranked_list.sort_evidence,
                                   ranked_list.candidates, ranked_list.source,
                                   ranked_list.order, ranked_list.sort_active)
            if selected_item is not None:
                self.confirm_selected(selected_item.snapshot_id, selected_item.visible_title, selected_item.source)
            if selected_item is None and not ordering_requirement(self.original_command) and self.candidate_selection:
                latest = self.observations.get(self.latest_snapshot, {})
                title = self.candidate_selection["title"]
                if normalized(title) in normalized(latest.get("text", "")):
                    self.confirm_selected(self.latest_snapshot, title, "ui")
            for check in checks:
                try:
                    self.record(check.criterion_id, "satisfied", check.evidence_text,
                                check.snapshot_id, check.source, user_answers, action_history)
                except ValueError as exc:
                    raise ValueError(f"{check.criterion_id} (관찰 {check.snapshot_id}, 출처 {check.source}): {exc}") from exc
        except Exception:
            self.criteria = before
            raise
        return self.missing()

    def missing(self):
        if not self.ready:
            return ["set_task_plan으로 원래 목표와 완료 기준부터 기록"]
        missing = [f"{key}: {item['description']}" for key, item in self.criteria.items()
                   if item["status"] != "satisfied"]
        if ordering_requirement(self.original_command):
            if not self.ranked_selection:
                missing.append("정렬 근거와 첫 번째부터 요청 순번까지 후보를 record_ranked_candidates로 확인")
            if not self.selected_item_matches:
                missing.append("선택 후 현재 항목의 제목을 confirm_selected_item으로 대조")
        if not ordering_requirement(self.original_command) and view_requirement(self.original_command):
            if not self.candidate_selection:
                missing.append("선택한 영상 카드의 실제 조회수 조건 확인")
            if not self.selected_item_matches:
                missing.append("조건에 맞는 후보와 현재 재생 제목을 confirm_selected_item으로 대조")
        return missing

    def state(self):
        return {"original_command": self.original_command, "goal": self.goal,
                "constraints": self.constraints, "steps": self.steps, "criteria": self.criteria,
                "ordering_requirement": ordering_requirement(self.original_command),
                "ranked_selection": self.ranked_selection,
                "view_requirement": view_requirement(self.original_command),
                "candidate_selection": self.candidate_selection,
                "selected_item_matches": self.selected_item_matches,
                "observed_app_identities": {package: sorted(names) for package, names in self.app_names.items()},
                "recent_plan_revisions": self.revisions[-3:],
                "observation_index": [
                    {"snapshot_id": key, "package": value["package"], "image_seen": value["image_seen"],
                     "text_excerpt": value["text"][:700],
                     "video_cards": [{"text": label, "views": displayed_views(label)} for label in value.get("cards", [])[:8]]}
                    for key, value in list(self.observations.items())[-16:]]}
