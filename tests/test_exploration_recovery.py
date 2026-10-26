"""History suggestions, vision-only rejected candidates, and observation-loop recovery."""
from contextlib import redirect_stdout
import io
import time
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.workflows import SearchVideoCard, YOUTUBE_PACKAGE, youtube_search_suggestion
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

COMMAND="유튜브에서 아이언맨 조회수 100만 이상인 영상 찾아줘"
LOW="아이언맨 예고편 - 3분 1초 - 조회수 90만회 - 동영상 재생"

class ExplorationRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice()
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.device.screen=youtube_screen(["최근 검색","아이언맨"])
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND
  self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("조건 영상 찾기",[],["조건 확인"],["검색","결과 탐색"])
  self.session.youtube_search_scope="topic"
  self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def test_history_click_without_typing_records_query_and_result_cards(self):
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([LOW]))
  with redirect_stdout(io.StringIO()):self.tools["click_element"].invoke({"snapshot_id":1,"element_id":"1"})
  memory=self.session.youtube_search_memory
  self.assertEqual(memory.searches,["아이언맨"])
  self.assertEqual(memory.state(COMMAND)["candidates_examined_for_query"],1)
  self.assertFalse(memory.state(COMMAND)["visible_candidates"][0]["views_match"])
  self.assertFalse(self.session.youtube_channel_pending)
 def test_plain_title_on_results_is_not_mistaken_for_history_suggestion(self):
  page=youtube_screen(["아이언맨"])
  self.assertIsNone(youtube_search_suggestion(page,page["elements"][0],COMMAND))
 def test_rejected_image_candidate_can_be_remembered_without_approving_it(self):
  self.session.goal_tracker.remember(1,self.device.screen,image_seen=True)
  result=self.tools["record_youtube_search_page"].invoke({"snapshot_id":1,"query":"아이언맨","source":"image", "candidates":[{"title":"아이언맨 예고편","view_count_text":"조회수 90만회"}]})
  self.assertFalse(result["search_memory"]["visible_candidates"][0]["views_match"])
  self.assertIsNone(self.session.goal_tracker.candidate_selection)
  self.assertFalse(self.session.completed)
 def test_image_candidate_requires_actual_image_observation(self):
  with self.assertRaises(ValueError):self.tools["record_youtube_search_page"].invoke({"snapshot_id":1,"query":"아이언맨","source":"image", "candidates":[{"title":"아이언맨","view_count_text":"조회수 90만회"}]})
 def test_duplicate_search_is_blocked_before_tap_until_exploration(self):
  memory=self.session.youtube_search_memory
  memory.submit("아이언맨");memory.submit("아이언맨")
  memory.record_page("아이언맨",[SearchVideoCard(title="예고편",view_count_text="조회수 90만회")])
  with self.assertRaisesRegex(ValueError,"같은 검색어"):
   self.tools["click_element"].invoke({"snapshot_id":1,"element_id":"1"})
  self.device.tap.assert_not_called()
  self.assertEqual(self.session.actions,0)
  memory.observe(youtube_screen([LOW]),1,"스크롤: down")
  self.session.check_youtube_query("아이언맨")
 def test_three_observations_offer_action_tools_instead_of_another_observation(self):
  self.session.observe();self.session.observe()
  self.assertTrue(self.session.exploration_due())
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["새 후보",LOW]))
  model=ToolRecordingModel(responses=[call("scroll_screen",{"direction":"down"},1),call("finish_task",{"summary":"탐색 중","goal_completed":False},2)])
  # Exercise middleware once without asking the scripted model to finish the task.
  from mobile_jarvis_app.agent import build_agent
  agent=build_agent(self.session,model=model)
  with redirect_stdout(io.StringIO()):
   agent.invoke({"messages":[{"role":"user","content":COMMAND}]})
  self.assertNotIn("observe_screen",model.available[0])
  for name in ("scroll_screen","swipe_region","click_element","type_text","navigate"):
   self.assertIn(name,model.available[0])
  self.assertIn("observe_screen",model.available[1])
  self.device.scroll.assert_called_once_with("down")
 def test_old_image_does_not_block_required_new_observation(self):
  self.session.screen={**self.device.screen,"elements":[]}
  self.session.observations_without_action=3
  self.session.visual_snapshot={"snapshot_id":self.session.snapshot_id,"captured_at":time.monotonic()-61}
  self.assertFalse(self.session.exploration_due())
 def test_finalization_is_not_subject_to_exploration_gate(self):
  self.session.observe();self.session.observe();self.session.finalization_pending=True
  self.assertFalse(self.session.exploration_due())
