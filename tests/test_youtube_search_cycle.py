"""Replay the user's back/search cycle with empty candidate memory."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_tools, build_agent
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

COMMAND="유튜브에서 아이언맨 조회수 100만 이상 영상 아무거나 재생해줘"

class SearchCycleTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen(["아이언맨","검색"])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND
  self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("조건 영상 재생",[],["영상 재생"],["검색","후보 탐색"])
  self.session.youtube_search_scope="topic"
  names=["클릭: 검색","클릭: 아이언맨","이동: back","클릭: 검색","텍스트 입력", "이미지 클릭: 키보드 우측 하단 돋보기 검색 버튼","클릭: 아이언맨","이동: back"]
  self.session.action_history=[{"action":name,"status":"executed","strategy":"navigate" if "이동" in name else "click"} for name in names]
  self.session.actions=len(names)
  self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def assess(self,kind="results",correct=True):
  return self.tools["assess_youtube_screen"].invoke({"snapshot_id":self.session.snapshot_id,"screen_kind":kind,"query_correct":correct,"target_visible":False,"evidence_text":"아이언맨","source":"ui"})
 def test_cycle_detected_without_query_or_candidate_records_and_survives_replan(self):
  self.assertEqual(self.session.youtube_search_memory.candidates,{})
  self.assertTrue(self.session.youtube_cycle_state()["recovery_due"])
  with redirect_stdout(io.StringIO()):self.session.autonomous_replan("같은 화면 세 번")
  self.assertTrue(self.session.youtube_cycle_state()["recovery_due"])
  self.assertEqual(self.session.youtube_cycle_state()["back_count"],2)
 def test_model_must_assess_then_can_choose_scroll_instead_of_back_loop(self):
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["새 후보","조회수 90만회"]))
  model=ToolRecordingModel(responses=[
   call("assess_youtube_screen",{"snapshot_id":1,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"아이언맨","source":"ui"},1),
   call("scroll_screen",{"direction":"down"},2),
   call("finish_task",{"summary":"탐색 중","goal_completed":False},3)])
  with redirect_stdout(io.StringIO()):build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":COMMAND}]})
  self.assertEqual(model.available[0],["assess_youtube_screen"])
  self.assertIn("scroll_screen",model.available[1])
  self.device.scroll.assert_called_once_with("down")
  self.device.navigate.assert_not_called()
 def test_assessed_correct_results_cannot_be_abandoned_immediately(self):
  self.assess()
  before=self.session.actions
  with self.assertRaisesRegex(ValueError,"결과 목록"):
   self.tools["navigate"].invoke({"direction":"back"})
  self.device.navigate.assert_not_called()
  self.assertEqual(self.session.actions,before)
 def test_wrong_query_or_editor_can_still_go_back(self):
  self.assess("search_editor")
  with redirect_stdout(io.StringIO()):self.tools["navigate"].invoke({"direction":"back"})
  self.device.navigate.assert_called_once_with("back")
 def test_duplicate_submission_is_tracked_for_keyboard_image_search(self):
  self.session.youtube_query="아이언맨";self.session.youtube_query_pending=True
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["검색 결과"]))
  with redirect_stdout(io.StringIO()):self.session.perform("이미지 클릭: 키보드 우측 하단 돋보기 검색 버튼",lambda:self.device.tap(50,50))
  self.assertEqual(self.session.youtube_search_memory.searches,["아이언맨"])
 def test_two_searches_without_cards_need_assessed_editor_for_retry(self):
  memory=self.session.youtube_search_memory
  memory.submit("아이언맨");memory.submit("아이언맨")
  with self.assertRaises(ValueError):self.session.check_youtube_query("아이언맨")
  self.assess("search_editor")
  self.session.check_youtube_query("아이언맨")
 def test_stale_assessment_cannot_block_a_new_screen(self):
  self.assess()
  self.session.observe()
  self.assertIsNone(self.session.current_youtube_assessment())
 def test_new_candidates_reset_progress_not_merely_navigation(self):
  memory=self.session.youtube_search_memory;memory.submit("아이언맨")
  self.device.screen=youtube_screen(["아이언맨 영상 - 2분 1초 - 조회수 90만회"])
  self.session.observe()
  self.assertEqual(memory.progress_action_id,len(self.session.action_history))
  self.assertFalse(self.session.youtube_cycle_state()["recovery_due"])
