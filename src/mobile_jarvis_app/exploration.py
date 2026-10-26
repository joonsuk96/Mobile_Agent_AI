"""App-independent exploration memory. No app names, tasks or fixed action routes."""
from collections import Counter
import hashlib
import re


class ExplorationMemory:
    def __init__(self):
        self.progress_action_id = 0
        self.fact_keys = set()
        self.assessment = None
        self.assessed_cycle = None
        self.assessment_failures = 0
        self.failure_cycle = None
        self.force_image_refresh = False
        self.assessment_checkpoint = 0

    def assessment_failed(self, cycle):
        if cycle != self.failure_cycle:
            self.failure_cycle = cycle
            self.assessment_failures = 0
        self.assessment_failures += 1
        if self.assessment_failures == 1:
            self.force_image_refresh = True
            return "refresh_image"
        self.release_assessment(cycle)
        return "release_actions"

    def release_assessment(self, cycle):
        # A failure to classify the screen is not goal failure or success.
        # Release only this bookkeeping gate; preserve action/call budgets.
        self.assessed_cycle = cycle
        self.force_image_refresh = False

    def assessment_succeeded(self, cycle, action_id=0):
        self.assessed_cycle = cycle
        self.assessment_checkpoint = action_id
        self.assessment_failures = 0
        self.failure_cycle = None
        self.force_image_refresh = False

    def record_fact(self, package, text, action_id):
        normalized = "".join(text.casefold().split())
        key = hashlib.sha256((package + ":" + normalized).encode()).hexdigest()
        if not normalized or key in self.fact_keys:
            return False
        self.fact_keys.add(key)
        self.progress_action_id = action_id
        return True

    def state(self, history, *, progress_floor=0, enabled=True):
        progress = max(self.progress_action_id, progress_floor)
        all_actions = [item.get("action", "") for item in history if item.get("status") == "executed"]
        window_start = max(progress, self.assessment_checkpoint)
        actions = [item.get("action", "") for item in history[window_start:] if item.get("status") == "executed"]
        backs = sum(action == "이동: back" for action in all_actions)
        recent_backs = actions.count("이동: back")
        searches = sum(bool(re.search(r"검색|search|돋보기", action, re.IGNORECASE))
                       and not re.search(r"필터|filter|정렬|옵션", action, re.IGNORECASE) for action in actions)
        token = None
        repeated = []
        if recent_backs >= 2 and searches >= 2:
            token = ("back_search", backs, searches)
        else:
            tail = actions[-18:]
            # Repeated navigation sequences, not a ban on consecutive scrolling.
            for size in range(2, min(6, len(tail)//2) + 1):
                block = tail[-size:]
                if tail[-2*size:-size] == block and any(not action.startswith(("스크롤:", "영역 스와이프")) for action in block):
                    repeated = block
                    token = ("sequence", tuple(block), sum(Counter(tail).values()))
                    break
        return {"recovery_due": bool(enabled and token and token != self.assessed_cycle),
                "cycle_token": token, "repeated_actions": repeated,
                "back_count": backs, "backs_since_new_candidates": recent_backs,
                "search_actions_since_new_candidates": searches,
                "new_candidates_at_action": progress, "progress_action_id": progress, "recovery_window_start": window_start,
                "distinct_facts": len(self.fact_keys),
                "principle": "새 항목/사실/조건 확인이 진전이다. back·입력·클릭만으로 실패 이력을 지우지 않는다."}
