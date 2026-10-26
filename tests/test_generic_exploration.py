"""App-independent exploration: arbitrary packages and task text, no service rules."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.exploration import ExplorationMemory
from test_agent import FakeDevice, FakeCatalog, screen, call
from test_app_launch_routing import ToolRecordingModel

class ExplorationMemoryTests(unittest.TestCase):
 def test_navigation_loop_without_search_words_is_detected(self):
  memory=ExplorationMemory()
  history=[{"action":action,"status":"executed"} for action in ["클릭: 메뉴","이동: back"]*2]
  self.assertTrue(memory.state(history)["recovery_due"])
 def test_consecutive_scroll_is_not_a_navigation_loop(self):
  memory=ExplorationMemory()
  history=[{"action":"스크롤: down","status":"executed"} for _ in range(8)]
  self.assertFalse(memory.state(history)["recovery_due"])
 def test_old_fact_does_not_reset_new_loop(self):
  memory=ExplorationMemory()
  self.assertTrue(memory.record_fact("any.app","대상 후보 A",4))
  self.assertFalse(memory.record_fact("any.app","대상 후보 A",8))
  self.assertEqual(memory.progress_action_id,4)
  history=[{"action":action,"status":"executed"} for action in ["클릭: 메뉴","이동: back"]*4]
  self.assertTrue(memory.state(history)["recovery_due"])

class GenericAgentRecoveryTests(unittest.TestCase):
 def session(self,package):
  device=FakeDevice();device.screen=screen(package,"목록의 후보 A")
  device.current_app=Mock(return_value={"package":package,"activity":package+"/.Main"})
  session=PhoneSession(device,FakeCatalog())
  session.original_command="원하는 항목을 찾아서 열어줘"
  session.goal_tracker.set_plan("항목 열기",[],["목표 항목 확인"],["항목 탐색"])
  session.action_history=[{"action":action,"status":"executed","strategy":"click"} for action in ["클릭: 메뉴","이동: back"]*2]
  session.actions=4;session.observe()
  return session,device,{tool.name:tool for tool in build_tools(session)}
 def test_same_recovery_policy_applies_to_arbitrary_apps(self):
  for package in ["org.example.bus","org.example.stocks","org.example.browser"]:
   with self.subTest(package=package):
    session,device,tools=self.session(package)
    self.assertTrue(session.navigation_cycle_state()["recovery_due"])
    model=ToolRecordingModel(responses=[
     call("assess_current_screen",{"snapshot_id":1,"screen_kind":"list","context_correct":True,"target_visible":False,"evidence_text":"목록의 후보 A","source":"ui"},1),
     call("scroll_screen",{"direction":"down"},2),
     call("finish_task",{"summary":"탐색 중","goal_completed":False},3)])
    device.scroll.side_effect=lambda *args:setattr(device,"screen",screen(package,"목록의 후보 B"))
    with patch("mobile_jarvis_app.agent.time.sleep"),redirect_stdout(io.StringIO()):
     build_agent(session,model=model).invoke({"messages":[{"role":"user","content":session.original_command}]})
    self.assertEqual(model.available[0],["assess_current_screen"])
    device.scroll.assert_called_once_with("down");device.navigate.assert_not_called()
 def test_common_assessment_protects_correct_list_and_allows_wrong_route_back(self):
  session,device,tools=self.session("arbitrary.package")
  args={"snapshot_id":1,"screen_kind":"list","context_correct":True,"target_visible":False,"evidence_text":"목록의 후보 A","source":"ui"}
  with redirect_stdout(io.StringIO()):tools["assess_current_screen"].invoke(args)
  with self.assertRaises(ValueError):tools["navigate"].invoke({"direction":"back"})
  args["context_correct"]=False
  with patch("mobile_jarvis_app.agent.time.sleep"),redirect_stdout(io.StringIO()):
   tools["assess_current_screen"].invoke(args)
   tools["navigate"].invoke({"direction":"back"})
  device.navigate.assert_called_once_with("back")
 def test_new_grounded_facts_reset_loop_but_duplicate_or_unsupported_facts_do_not(self):
  session,device,tools=self.session("arbitrary.package")
  with redirect_stdout(io.StringIO()):result=tools["record_exploration_progress"].invoke({"snapshot_id":1,"facts":["목록의 후보 A"],"source":"ui"})
  self.assertEqual(result["new_facts"],1)
  self.assertFalse(session.navigation_cycle_state()["recovery_due"])
  with redirect_stdout(io.StringIO()):result=tools["record_exploration_progress"].invoke({"snapshot_id":1,"facts":["목록의 후보 A"],"source":"ui"})
  self.assertEqual(result["new_facts"],0)
  with self.assertRaises(ValueError):tools["record_exploration_progress"].invoke({"snapshot_id":1,"facts":["없는 항목"],"source":"ui"})
 def test_replan_preserves_generic_recovery_and_finalization_disables_it(self):
  session,device,tools=self.session("arbitrary.package")
  with redirect_stdout(io.StringIO()):session.autonomous_replan("진전 없음")
  self.assertTrue(session.navigation_cycle_state()["recovery_due"])
  session.finalization_pending=True
  self.assertFalse(session.navigation_cycle_state()["recovery_due"])
