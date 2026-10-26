"""Interactive recovery tests: no phone, network, or paid model calls."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from mobile_jarvis_app.agent import (
    PhoneSession, TaskCancelled, pending_tool_repairs, read_user_guidance, run_interactive,
)
from mobile_jarvis_app.device import DeviceError
from test_agent import FakeCatalog, FakeDevice, ScriptedModel, call, screen


class RecordingModel(ScriptedModel):
    histories: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.histories.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FailOnceModel(RecordingModel):
    failed: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position == 1 and not self.failed:
            self.failed = True
            raise RuntimeError("private-request-sentinel")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.session = PhoneSession(self.device, FakeCatalog(), interactive=True, require_plan=False)
        delay = patch("mobile_jarvis_app.agent.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_model_giving_up_prompts_user_and_keeps_original_history(self):
        goal = "수원터미널을 찾아줘"
        model = RecordingModel(responses=[
            AIMessage(content="현재 화면에서 대상을 찾지 못했어."),
            call("ask_user", {"question": "목록에서 어디로 이동하면 될까?"}, 4),
            call("click_element", {"snapshot_id": 1, "element_id": "0"}, 2),
            call("finish_task", {"summary": "수원터미널을 찾았어."}, 3),
        ])
        def guidance(prompt):
            self.device.screen = screen(text="수원터미널")
            return "내가 경기 목록으로 옮겼어. 여기에서 수원터미널을 눌러줘."
        self.device.tap.side_effect = lambda *args: setattr(
            self.device, "screen", screen(text="수원터미널 상세"))
        with patch("builtins.input", side_effect=guidance) as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, goal, model=model)
        ask.assert_called_once()
        self.assertTrue(self.session.completed)
        self.device.tap.assert_called_once()
        resumed = model.histories[2]
        self.assertTrue(any(isinstance(m, HumanMessage) and m.content == goal for m in resumed))
        self.assertTrue(any(isinstance(m, AIMessage) and "찾지 못했어" in str(m.content) for m in resumed))
        self.assertTrue(any("경기 목록" in str(m.content) for m in resumed))
        self.assertTrue(any("수원터미널" in str(m.content) for m in resumed))

    def test_repetition_pause_does_not_reexecute_pending_action(self):
        self.session.observe()
        self.session.unchanged = 1
        model = RecordingModel(responses=[
            call("navigate", {"direction": "back"}, 1),
            call("finish_task", {"summary": "안내받은 화면에서 확인했어."}, 2),
        ])
        def guidance(prompt):
            self.device.screen = screen(text="수원터미널")
            return "직접 목록을 열었어. 지금 화면을 확인해."
        with patch("builtins.input", side_effect=guidance), redirect_stdout(io.StringIO()):
            run_interactive(self.session, "터미널 찾기", model=model)
        self.device.navigate.assert_called_once_with("back")
        self.assertEqual(self.session.actions, 1)
        self.assertEqual(self.session.unchanged, 0)
        self.assertTrue(self.session.completed)
        self.assertTrue(any(isinstance(m, ToolMessage) and "paused_for_user" in str(m.content)
                            for m in model.histories[-1]))

    def test_action_limit_waits_for_user_then_renews_bounded_allowance(self):
        self.session.max_actions = 1
        self.session.actions = 1
        self.session.observe()
        model = RecordingModel(responses=[
            call("click_element", {"snapshot_id": 1, "element_id": "0"}, 1),
            call("scroll_screen", {"direction": "down"}, 2),
            call("finish_task", {"summary": "대상을 찾았어."}, 3),
        ])
        self.device.scroll.side_effect = lambda *args: setattr(
            self.device, "screen", screen(text="수원터미널"))
        with patch("builtins.input", return_value="계속 아래로 찾아봐") as ask, \
             redirect_stdout(io.StringIO()):
            run_interactive(self.session, "터미널 찾기", model=model)
        ask.assert_called_once()
        self.device.tap.assert_not_called()
        self.device.scroll.assert_called_once()
        self.assertEqual(self.session.actions, 2)
        self.assertEqual(self.session.action_window_start, 1)
        self.assertEqual(self.session.public_screen()["actions_used"], 1)

    def test_model_limit_can_continue_in_same_thread_after_guidance(self):
        self.session.max_model_calls = 1
        model = RecordingModel(responses=[
            call("observe_screen", {}, 1),
            call("finish_task", {"summary": "확인했어."}, 2),
        ])
        with patch("builtins.input", return_value="계속 확인해줘") as ask, \
             redirect_stdout(io.StringIO()):
            run_interactive(self.session, "화면 확인", model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 2)
        ask.assert_called_once()

    def test_tool_limit_can_continue_without_replaying_blocked_tool(self):
        self.session.max_tool_calls = 1
        model = RecordingModel(responses=[
            call("observe_screen", {}, 1),
            call("observe_screen", {}, 2),
            call("finish_task", {"summary": "확인했어."}, 3),
        ])
        with patch("builtins.input", return_value="계속해") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "화면 확인", model=model)
        self.assertTrue(self.session.completed)
        self.assertEqual(model.position, 3)
        ask.assert_called_once()

    def test_explicit_cancel_exits_without_another_question(self):
        model = RecordingModel(responses=[call("ask_user", {"question": "어디를 찾을까요?"}, 1)])
        with patch("builtins.input", return_value="/cancel") as ask, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(TaskCancelled):
                run_interactive(self.session, "대상 찾기", model=model)
        ask.assert_called_once()
        self.assertFalse(self.session.completed)

    def test_api_failure_keeps_completed_steps_and_does_not_print_private_error(self):
        model = FailOnceModel(responses=[
            call("observe_screen", {}, 1),
            call("finish_task", {"summary": "확인했어."}, 2),
        ])
        output = io.StringIO()
        with patch("builtins.input", return_value="다시 이어서 해줘"), redirect_stdout(output):
            run_interactive(self.session, "화면 확인", model=model)
        self.assertTrue(self.session.completed)
        self.assertNotIn("private-request-sentinel", output.getvalue())

    def test_failed_observation_does_not_lose_user_guidance(self):
        self.device.read_screen = lambda: (_ for _ in ()).throw(DeviceError("폰 연결 확인"))
        result = self.session.resume_with_guidance("와이파이를 다시 연결했어")
        self.assertEqual(result["reply"], "와이파이를 다시 연결했어")
        self.assertIn("screen_error", result)
        self.assertIsNone(self.session.screen)

    def test_empty_reply_waits_locally_for_real_instruction(self):
        with patch("builtins.input", side_effect=["", "계속"]), redirect_stdout(io.StringIO()):
            self.assertEqual(read_user_guidance("어떻게 할까요?"), "계속")

    def test_partial_finish_continues_without_asking_user(self):
        self.session.observe()
        model = RecordingModel(responses=[
            call("finish_task", {"summary": "Google로 이동했지만 제목은 확인하지 못했어.", "goal_completed": False, "remaining_work": "검색 결과의 제목 확인"}, 1),
            call("observe_screen", {}, 2),
            call("finish_task", {"summary": "요청한 제목을 확인했어."}, 3),
        ])
        with patch("builtins.input") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "메이플인벤 10추 게시글 제목 확인", model=model)
        ask.assert_not_called()
        self.assertEqual(model.position, 3)
        self.assertTrue(any("검색 결과의 제목" in str(m.content) for m in model.histories[-1]))

    def test_success_finishes_without_any_confirmation(self):
        self.session.observe()
        model = RecordingModel(responses=[call("finish_task", {"summary": "확인했어"}, 1)])
        with patch("builtins.input") as ask, redirect_stdout(io.StringIO()):
            run_interactive(self.session, "화면 확인", model=model)
        ask.assert_not_called()
        self.assertTrue(self.session.completed)

    def test_repeated_partial_finish_eventually_requests_help(self):
        self.session.observe()
        model = RecordingModel(responses=[
            call("finish_task", {"summary": "미완료", "goal_completed": False}, i)
            for i in range(1, 4)
        ])
        with patch("builtins.input", return_value="/cancel") as ask, redirect_stdout(io.StringIO()):
            with self.assertRaises(TaskCancelled):
                run_interactive(self.session, "검색", model=model)
        ask.assert_called_once()
        self.assertEqual(model.position, 3)

    def test_plain_view_count_answer_finalizes_without_asking_or_repeating_actions(self):
        self.session.observe()
        model = RecordingModel(responses=[
            AIMessage(content="두 번째 최근 영상의 현재 조회수는 **7.5만 회**입니다."),
            call("finish_task", {"summary": "두 번째 최근 영상의 현재 조회수는 **7.5만 회**입니다."}, 2),
        ])
        output = io.StringIO()
        with patch("builtins.input") as ask, redirect_stdout(output):
            run_interactive(self.session, "두 번째 최근 영상 조회수 알려줘", model=model)
        ask.assert_not_called()
        self.assertEqual(output.getvalue().count("7.5만"), 1)
        self.assertTrue(self.session.completed)
        self.device.tap.assert_not_called()
        self.assertEqual(self.session.actions, 0)

    def test_failed_checkpoint_repair_still_asks_user(self):
        from unittest.mock import Mock
        broken = Mock()
        broken.invoke.side_effect = RuntimeError("private-request-sentinel")
        broken.get_state.side_effect = RuntimeError("private-state-sentinel")
        with patch("mobile_jarvis_app.agent.build_agent", return_value=broken), \
             patch("builtins.input", return_value="/cancel") as ask, redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(TaskCancelled):
                run_interactive(self.session, "검색해줘")
        ask.assert_called_once()
        self.assertNotIn("private-", output.getvalue())

    def test_only_unfinished_tool_calls_are_repaired(self):
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "navigate", "args": {"direction": "back"}, "id": "done", "type": "tool_call"},
                {"name": "navigate", "args": {"direction": "home"}, "id": "pending", "type": "tool_call"},
            ]),
            ToolMessage(content="done", tool_call_id="done"),
        ]
        repairs = pending_tool_repairs(messages)
        self.assertEqual([message.tool_call_id for message in repairs], ["pending"])


if __name__ == "__main__":
    unittest.main()
