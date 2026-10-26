"""Classification metadata failures must not trap the agent in one forced tool."""
import base64
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.exploration import ExplorationMemory
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen
from test_browser_fallback import PNG
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE

class AssessmentEscapeTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen(["검색","아이언맨"])
  self.device.capture_screen=Mock(return_value={"image_url":"data:image/png;base64,"+base64.b64encode(PNG).decode(),"width":1,"height":1,"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command="유튜브에서 아이언맨 롱폼 조회수 100만 이상 찾아줘"
  self.session.goal_tracker.set_plan("영상 재생",[],["목표 확인"],["검색","탐색"])
  self.session.action_history=[{"action":action,"status":"executed"} for action in ["클릭: 검색","이동: back"]*2]
  self.session.actions=4;self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def run_script(self,responses):
  model=ToolRecordingModel(responses=responses)
  with redirect_stdout(io.StringIO()) as output:
   build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":self.session.original_command}]})
  return model,output.getvalue()
 def test_two_rejected_assessments_refresh_image_once_then_release_actions(self):
  bad={"snapshot_id":999,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"아이언맨","source":"ui"}
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["아래 결과"]))
  model,output=self.run_script([
   call("assess_youtube_screen",bad,1),call("observe_screen",{},2),
   call("assess_youtube_screen",bad,3),call("scroll_screen",{"direction":"down"},4),
   call("finish_task",{"summary":"탐색 중","goal_completed":False},5)])
  self.assertEqual(model.available[0],["assess_youtube_screen"])
  self.assertEqual(model.available[1],["observe_screen"])
  self.assertEqual(model.available[2],["assess_youtube_screen"])
  self.assertIn("scroll_screen",model.available[3])
  self.assertGreater(len(model.available[3]),1)
  self.device.capture_screen.assert_called_once()
  self.device.scroll.assert_called_once_with("down")
  self.assertIn("[판정 제한 해제]",output)
  self.assertFalse(self.session.completed)
 def test_schema_error_also_gets_refresh_instead_of_repeating_same_tool(self):
  valid={"snapshot_id":2,"screen_kind":"dialog","query_correct":True,"target_visible":False,"evidence_text":"필터 화면","source":"image"}
  model,output=self.run_script([
   call("assess_youtube_screen",{},1),call("observe_screen",{},2),
   call("assess_youtube_screen",valid,3),call("finish_task",{"summary":"탐색 중","goal_completed":False},4)])
  self.assertEqual(model.available[1],["observe_screen"])
  self.assertEqual(self.session.current_screen_assessment()["screen_kind"],"dialog")
  self.assertEqual(self.session.exploration.assessment_failures,0)
 def test_mislabeled_ui_paraphrase_uses_existing_image_not_invented_xml(self):
  self.session.goal_tracker.remember(1,self.device.screen,image_seen=True)
  with redirect_stdout(io.StringIO()):result=self.tools["assess_youtube_screen"].invoke({"snapshot_id":1,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"검색 결과에 일반 영상 후보가 보이지 않는 상태","source":"ui"})
  self.assertEqual(result["assessment"]["screen_kind"],"results")
  self.assertFalse(self.session.completed)
 def test_image_refresh_failure_also_releases_the_single_tool_gate(self):
  self.device.capture_screen.side_effect=ValueError("capture unavailable")
  bad={"snapshot_id":999,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"결과","source":"ui"}
  model,output=self.run_script([
   call("assess_youtube_screen",bad,1),call("observe_screen",{},2),
   call("finish_task",{"summary":"탐색 중","goal_completed":False},3)])
  self.assertGreater(len(model.available[2]),1)
  self.assertIn("이미지 보완 실패",output)
  self.assertFalse(self.session.exploration.force_image_refresh)
 def test_new_cycle_can_request_its_own_bounded_refresh(self):
  memory=ExplorationMemory()
  self.assertEqual(memory.assessment_failed(("cycle",1)),"refresh_image")
  self.assertEqual(memory.assessment_failed(("cycle",1)),"release_actions")
  self.assertEqual(memory.assessment_failed(("cycle",2)),"refresh_image")


class RecoveryWindowTests(unittest.TestCase):
 def test_valid_assessment_starts_new_recovery_window_without_claiming_progress(self):
  memory=ExplorationMemory()
  history=[{"action":action,"status":"executed"} for action in ["클릭: 검색","이동: back"]*2]
  first=memory.state(history)
  memory.assessment_succeeded(first["cycle_token"],len(history))
  history.append({"action":"클릭: 검색 필터","status":"executed"})
  state=memory.state(history)
  self.assertFalse(state["recovery_due"])
  self.assertEqual(state["progress_action_id"],0)
  self.assertEqual(state["distinct_facts"],0)
  history += [{"action":action,"status":"executed"} for action in ["클릭: 검색","이동: back"]*2]
  self.assertTrue(memory.state(history)["recovery_due"])
 def test_filter_clicks_are_not_counted_as_repeated_query_submissions(self):
  memory=ExplorationMemory()
  history=[{"action":action,"status":"executed"} for action in ["클릭: 검색 필터","이동: back","클릭: 길이 필터","이동: back"]]
  self.assertEqual(memory.state(history)["search_actions_since_new_candidates"],0)
