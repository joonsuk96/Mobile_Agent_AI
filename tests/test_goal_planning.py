"""Goal contracts, ranked selection and correction through actual agent tools."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.device import parse_screen
from mobile_jarvis_app.planning import GoalTracker, RankedCandidate, ordering_requirement
from test_agent import FakeCatalog, FakeDevice, call
from test_app_launch_routing import ToolRecordingModel
from test_interactive_recovery import RecordingModel


def view(labels):
    nodes = []
    for index, label in enumerate(labels):
        top = index * 30
        nodes.append(f'<node package="org.example.app" class="android.widget.TextView" enabled="true" clickable="true" text="{escape(label)}" bounds="[0,{top}][100,{top+25}]"/>')
    return parse_screen(("<hierarchy>" + "".join(nodes) + "</hierarchy>").encode(),
                        {"package": "org.example.app", "activity": "org.example.app/.Main"})


class GoalTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = GoalTracker("두 번째로 최근 항목 확인")
        self.tracker.set_plan("요청한 순번의 항목 확인", ["최신순 두 번째"], ["대상 확인", "결과 확인"], ["정렬 확인", "대상 선택"])
        self.list_view = view(["최신순", "첫 항목", "1일 전", "둘째 항목", "2일 전"])
        self.tracker.remember(1, self.list_view)
        self.candidates = [RankedCandidate(position=1, title="첫 항목", publication_text="1일 전"),
                           RankedCandidate(position=2, title="둘째 항목", publication_text="2일 전")]

    def test_order_requirement_handles_korean_and_typo_forms(self):
        for command in ["2번쨰로 최근 영상 재생", "2번째 최신 영상", "두 번째 최근 항목", "최신 영상 중 2번째 선택"]:
            with self.subTest(command=command):
                self.assertEqual(ordering_requirement(command), {"rank": 2, "order": "newest"})
        self.assertIsNone(ordering_requirement("삼성전자 현재가 확인"))

    def test_completion_rejects_unmet_criteria(self):
        self.assertIn("C1: 대상 확인", self.tracker.missing())

    def test_ui_evidence_must_exist_in_observed_snapshot(self):
        with self.assertRaises(ValueError):
            self.tracker.record("C1", "satisfied", "없는 제목", 1, "ui", {})
        self.tracker.record("C1", "satisfied", "둘째 항목", 1, "ui", {})
        self.assertEqual(self.tracker.criteria["C1"]["status"], "satisfied")

    def test_image_evidence_requires_an_actual_image_observation(self):
        with self.assertRaises(ValueError):
            self.tracker.record("C1", "satisfied", "이미지에서 읽은 제목", 1, "image", {})
        self.tracker.remember(2, view([]), image_seen=True)
        self.tracker.record("C1", "satisfied", "이미지에서 읽은 제목", 2, "image", {})
        self.assertEqual(self.tracker.criteria["C1"]["evidence"]["source"], "image")

    def test_plan_revision_cannot_change_original_goal_or_criteria(self):
        old_criteria = dict(self.tracker.criteria)
        self.tracker.revise("잘못된 목록", ["뒤로 이동", "정렬된 목록 열기"])
        self.assertEqual(self.tracker.criteria, old_criteria)
        self.assertEqual(self.tracker.original_command, "두 번째로 최근 항목 확인")
        with self.assertRaises(ValueError):
            self.tracker.set_plan("아무 항목", [], ["열기"], ["클릭"])

    def test_ranked_candidates_require_full_prefix_and_observed_sort(self):
        with self.assertRaises(ValueError):
            self.tracker.record_ranked(1, "인기순", self.candidates, "ui")
        with self.assertRaises(ValueError):
            self.tracker.record_ranked(1, "최신순", self.candidates[1:], "ui")
        selected = self.tracker.record_ranked(1, "최신순", self.candidates, "ui")
        self.assertEqual(selected["title"], "둘째 항목")

    def test_visible_popular_sort_does_not_satisfy_latest_requirement(self):
        self.tracker.remember(2, view(["인기순", "첫 항목", "1일 전", "둘째 항목", "2일 전"]))
        with self.assertRaises(ValueError):
            self.tracker.record_ranked(2, "인기순", self.candidates, "ui", "newest", True)
        with self.assertRaises(ValueError):
            self.tracker.record_ranked(2, "최신순", self.candidates, "ui", "oldest", True)
        with self.assertRaises(ValueError):
            self.tracker.record_ranked(1, "최신순", self.candidates, "ui", "newest", False)

    def test_wrong_selection_is_detected_and_newer_observation_requires_reconfirmation(self):
        self.tracker.record_ranked(1, "최신순", self.candidates, "ui")
        self.tracker.remember(2, view(["첫 항목", "일시정지"]))
        result = self.tracker.confirm_selected(2, "첫 항목", "ui")
        self.assertFalse(result["matches"])
        self.assertIn("뒤로", result["next_step"])
        self.tracker.remember(3, view(["둘째 항목", "일시정지"]))
        self.assertTrue(self.tracker.confirm_selected(3, "둘째 항목", "ui")["matches"])
        self.tracker.remember(4, view(["첫 항목", "일시정지"]))
        self.assertTrue(any("confirm_selected_item" in value for value in self.tracker.missing()))

    def test_user_evidence_must_come_from_real_answer(self):
        with self.assertRaises(ValueError):
            self.tracker.record("C1", "satisfied", "성인 2명", 0, "user", {"인원": "성인 1명"})
        self.tracker.record("C1", "satisfied", "성인 1명", 0, "user", {"인원": "성인 1명"})


class GoalAgentTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog())
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_first_model_call_only_exposes_plan_tool_then_completion_needs_evidence(self):
        self.device.screen = view(["요청 항목"])
        model = ToolRecordingModel(responses=[
            call("set_task_plan", {"goal": "요청 항목 확인", "constraints": [], "success_criteria": ["항목 존재 확인"], "steps": ["화면 읽기"]}, 1),
            call("observe_screen", {}, 2),
            call("finish_task", {"summary": "확인했어"}, 3),
            call("record_task_progress", {"criterion_id": "C1", "status": "satisfied", "evidence_text": "요청 항목", "snapshot_id": 1}, 4),
            call("finish_task", {"summary": "요청 항목을 확인했어"}, 5),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "요청 항목 확인", model=model)
        self.assertEqual(model.available[0], ["set_task_plan"])
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 5)

    def test_physical_action_before_plan_is_rejected(self):
        tools = {t.name: t for t in build_tools(self.session)}
        with self.assertRaises(ValueError):
            tools["navigate"].invoke({"direction": "back"})
        self.device.navigate.assert_not_called()

    def test_full_planning_mode_pauses_for_missing_user_fact_and_preserves_goal(self):
        model = RecordingModel(responses=[
            call("set_task_plan", {"goal": "성인/아동 정보 확인", "constraints": ["종류는 사용자에게 확인"], "success_criteria": ["승객 구분 확인"], "steps": ["필수 정보 질문"]}, 1),
            call("request_missing_info", {"field": "승객 구분", "question": "성인인가요 아동인가요?", "why_required": "좌석 요금 구분에 필요"}, 2),
            call("record_task_progress", {"criterion_id": "C1", "status": "satisfied", "source": "user", "evidence_text": "성인 1명"}, 3),
            call("finish_task", {"summary": "성인 1명으로 확인했어"}, 4),
        ])
        with patch("builtins.input", return_value="성인 1명") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "승객 구분을 확인해줘", model=model)
        ask.assert_called_once()
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.goal_tracker.original_command, "승객 구분을 확인해줘")
        self.assertEqual(self.session.goal_tracker.criteria["C1"]["evidence"]["source"], "user")

    def test_wrong_ranked_item_goes_back_revises_plan_and_selects_correct_item(self):
        listing = view(["최신순", "첫 항목", "1일 전", "둘째 항목", "2일 전"])
        self.device.screen = listing
        def tap(x, y):
            title = "첫 항목" if y < 70 else "둘째 항목"
            self.device.screen = view([title, "일시정지"])
        self.device.tap.side_effect = tap
        self.device.navigate.side_effect = lambda direction: setattr(self.device, "screen", listing)
        model = RecordingModel(responses=[
            call("set_task_plan", {"goal": "두 번째 최신 항목 재생", "constraints": ["최신순 두 번째"], "success_criteria": ["요청 항목 제목 확인", "실제 재생 확인"], "steps": ["최신순 확인", "두 번째 항목 선택", "재생 확인"]}, 1),
            call("observe_screen", {}, 2),
            call("record_ranked_candidates", {"snapshot_id": 1, "order": "newest", "sort_active": True, "sort_evidence": "최신순", "candidates": [{"position": 1, "title": "첫 항목", "publication_text": "1일 전"}, {"position": 2, "title": "둘째 항목", "publication_text": "2일 전"}]}, 3),
            call("click_element", {"snapshot_id": 1, "element_id": "1"}, 4),
            call("confirm_selected_item", {"snapshot_id": 2, "visible_title": "첫 항목"}, 5),
            call("revise_task_plan", {"reason": "첫 항목을 잘못 선택함", "steps": ["뒤로 이동", "둘째 항목 선택", "제목과 재생 확인"]}, 6),
            call("navigate", {"direction": "back"}, 7),
            call("click_element", {"snapshot_id": 3, "element_id": "3"}, 8),
            call("confirm_selected_item", {"snapshot_id": 4, "visible_title": "둘째 항목"}, 9),
            call("record_task_progress", {"criterion_id": "C1", "status": "satisfied", "evidence_text": "둘째 항목", "snapshot_id": 4}, 10),
            call("record_task_progress", {"criterion_id": "C2", "status": "satisfied", "evidence_text": "일시정지", "snapshot_id": 4}, 11),
            call("finish_task", {"summary": "두 번째 최신 항목을 재생했어"}, 12),
        ])
        with patch("builtins.input", side_effect=AssertionError("must not ask")), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "두 번째로 최근 항목 재생해줘", model=model)
        self.device.navigate.assert_called_once_with("back")
        self.assertEqual(self.device.tap.call_count, 2)
        self.assertTrue(self.session.completed)
        self.assertEqual(self.session.goal_tracker.ranked_selection["title"], "둘째 항목")
        self.assertEqual(len(self.session.goal_tracker.revisions), 1)
        systems = [m.content for m in model.histories[5] if m.type == "system"]
        self.assertTrue(any("task_contract" in text and "두 번째로 최근 항목" in text for text in systems))
