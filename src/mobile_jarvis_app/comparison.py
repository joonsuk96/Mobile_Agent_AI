"""Local evidence checks for a maximum-view search within a calendar window.

No network, model invocation or device I/O. Coverage assertions still depend on
what the agent observes; they must never be inferred from a scroll count.
"""
import calendar
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from pydantic import BaseModel, Field
from .media import displayed_views


def comparison_requirement(command):
    if not re.search(r"조회수|views", command, re.I):
        return None
    if not re.search(r"(?:가장|제일).{0,14}(?:높|많)|최다|최고\s*조회수|최대\s*조회수|most\s+viewed", command, re.I):
        return None
    match = re.search(r"(?:최근|지난|지난번)?\s*(\d+)\s*(개월|달|주|일)\s*(?:이?내|동안|간)?", command)
    if not match:
        return None
    return {"amount": int(match[1]), "unit": match[2]}


def shift_months(day, months):
    index = day.year * 12 + day.month - 1 - months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def cutoff_date(requirement, today):
    amount, unit = requirement["amount"], requirement["unit"]
    return (shift_months(today, amount) if unit in {"개월", "달"}
            else today - timedelta(days=amount * (7 if unit == "주" else 1)))


def publication_interval(text, today):
    exact = re.search(r"(20\d{2})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2})", text)
    if exact:
        day = date(*map(int, exact.groups()))
        return day, day
    relative = re.search(r"(\d+)\s*(개월|달|주|일|시간|분)\s*전", text)
    if not relative:
        raise ValueError("업로드 날짜를 해석할 수 없어. 실제 정확한 날짜를 확인해.")
    amount, unit = int(relative[1]), relative[2]
    if unit in {"개월", "달"}:
        return shift_months(today, amount + 1), shift_months(today, amount)
    days = amount * (7 if unit == "주" else 1) if unit in {"주", "일"} else 0
    width = 7 if unit == "주" else 1
    return today - timedelta(days=days + width), today - timedelta(days=days)


class ComparisonCandidate(BaseModel):
    title: str = Field(min_length=1)
    views_text: str = Field(min_length=1)
    publication_text: str = Field(min_length=1)
    snapshot_id: int
    source: Literal["ui", "image"] = "ui"


class PeriodComparisonEvidence(BaseModel):
    sort_snapshot_id: int
    sort_evidence: str = Field(min_length=1, description="실제 활성화된 최신순 표시")
    sort_source: Literal["ui", "image"] = "ui"
    started_at_newest: bool = Field(description="최신순 맨 위부터 시작했는가")
    continuous_coverage: bool = Field(description="중간 후보나 페이지를 건너뛰지 않았는가")
    all_candidates_recorded: bool = Field(description="각 화면의 기간 내 후보를 빠짐없이 기록했는가")
    candidates: list[ComparisonCandidate] = Field(description="최신순으로 본 모든 후보. 마지막은 기간 밖 경계 후보여야 함")
    list_end_snapshot_id: int | None = None
    list_end_evidence: str = Field(default="", description="기간 밖 후보가 없으면 실제 목록 끝을 확인한 화면 근거")
    list_end_source: Literal["ui", "image"] = "ui"
    selected_title: str = Field(min_length=1)


class ComparisonEvidenceError(ValueError):
    """Evidence reference is wrong; repair metadata without restarting discovery."""


class ComparisonSortMissing(ValueError):
    """Only the active sort needs inspection in the current list."""


def check_recorded_evidence(tracker, snapshot_id, source, texts, *, sort=False):
    # Try the submitted frame first, then exact UI witnesses from the same app.
    # Candidate fields must all come from ONE frame, never mix unrelated cards.
    first = tracker.observations.get(snapshot_id, {})
    witnesses = [snapshot_id]
    if first.get("package"):
        witnesses += [key for key, value in tracker.observations.items()
                      if key != snapshot_id and value.get("package") == first["package"]]
    for key in witnesses:
        try:
            for text in texts:
                tracker.evidence(key, source if key == snapshot_id else "ui", text)
            return
        except ValueError:
            pass
    error = ComparisonSortMissing if sort else ComparisonEvidenceError
    raise error("현재 목록에서 활성 최신순 표시만 확인해. 후보를 다시 찾거나 채널에 재진입하지 마."
                if sort else f"관찰 {snapshot_id}의 후보 근거 연결을 기존 기록으로 보완해. 화면을 재탐색하지 마.")


def validate_period_comparison(tracker, proof, summary, today=None):
    requirement = comparison_requirement(tracker.original_command)
    if requirement is None:
        return
    if proof is None:
        raise ValueError("기간 전체 비교 근거가 없어. 최신순 맨 위부터 기간 밖 경계까지 후보의 제목·업로드 날짜·조회수를 비교해.")
    if not (proof.started_at_newest and proof.continuous_coverage and proof.all_candidates_recorded):
        raise ValueError("일부 화면만 확인했어. 누락된 후보와 기간 경계를 확인한 뒤 완료해.")
    if not re.search(r"최신|newest|latest", proof.sort_evidence, re.I):
        raise ComparisonSortMissing("현재 목록에서 실제 활성 최신순 표시만 확인해. 채널에 재진입하지 마.")
    check_recorded_evidence(tracker, proof.sort_snapshot_id, proof.sort_source, [proof.sort_evidence], sort=True)
    today = today or datetime.now(timezone(timedelta(hours=9))).date()
    cutoff = cutoff_date(requirement, today)
    eligible, seen, boundary = [], set(), False
    previous_upper = None
    from .planning import normalized
    for item in proof.candidates:
        if boundary:
            raise ValueError("기간 밖 경계 뒤에 후보가 있어. 실제 최신순 목록을 확인해.")
        key = normalized(item.title)
        if key in seen:
            raise ValueError("비교 목록에 중복 후보가 있어.")
        seen.add(key)
        check_recorded_evidence(tracker, item.snapshot_id, item.source,
                                [item.title, item.views_text, item.publication_text])
        low, high = publication_interval(item.publication_text, today)
        if high > today:
            raise ValueError("미래 업로드 날짜를 비교 근거로 사용할 수 없어.")
        if previous_upper is not None and low > previous_upper:
            raise ValueError("후보의 날짜가 최신순과 맞지 않아. 정렬과 누락 여부를 확인해.")
        previous_upper = high
        if high < cutoff:
            boundary = True
            continue
        if low < cutoff:
            raise ValueError("기간 경계의 상대 날짜가 모호해. 해당 후보의 정확한 업로드 날짜를 확인해.")
        views = displayed_views(item.views_text)
        if not views or views["value"] is None:
            raise ValueError("비교 후보의 실제 조회수 표시를 확인해.")
        eligible.append((item, views["value"]))
    if not boundary:
        if proof.list_end_snapshot_id is None or not proof.list_end_evidence.strip():
            raise ValueError("아직 기간 밖 경계나 실제 목록 끝까지 확인하지 않았어. 최신순 목록 탐색을 계속해.")
        check_recorded_evidence(tracker, proof.list_end_snapshot_id, proof.list_end_source, [proof.list_end_evidence])
        if proof.list_end_snapshot_id < max((item.snapshot_id for item in proof.candidates), default=0):
            raise ValueError("마지막 후보 이후의 실제 목록 끝 근거가 필요해.")
    if not eligible:
        raise ValueError("기간 내 비교 후보가 없어.")
    maximum = max(value for _, value in eligible)
    winners = [item for item, value in eligible if value == maximum]
    selected = next((item for item in winners if normalized(item.title) == normalized(proof.selected_title)), None)
    if selected is None:
        raise ValueError("선택 제목이 기간 내 확인한 최대 조회수 후보와 달라.")
    if normalized(selected.title) not in normalized(summary) or normalized(selected.views_text) not in normalized(summary):
        raise ValueError("최종 답변에 최대 조회수 후보의 실제 제목과 조회수 표시를 포함해.")
