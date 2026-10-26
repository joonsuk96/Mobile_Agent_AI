"""Automatic fresh screenshots and compact inputs remove redundant observation calls."""
import base64
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock,patch
from pydantic import Field
from langchain.messages import ToolMessage
from mobile_jarvis_app.agent import PhoneSession,build_agent,build_tools
from mobile_jarvis_app.context_window import compact_observation_history
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice,FakeCatalog,call,screen
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen
from test_browser_fallback import PNG

class InputModel(ToolRecordingModel):
 inputs:list=Field(default_factory=list)
 def _generate(self,messages,**kwargs):
  self.inputs.append(messages)
  return super()._generate(messages,**kwargs)

class ObservationLatencyTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen(["아이언맨 검색 결과"])
  self.device.read_screen=Mock(side_effect=lambda:self.device.screen)
  self.device.capture_screen=Mock(return_value={"image_url":"data:image/png;base64,"+base64.b64encode(PNG).decode(),"width":1,"height":1,"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command="유튜브에서 아이언맨 영상 찾아줘"
  self.session.goal_tracker.set_plan("영상 찾기",[],["목표 확인"],["목록 탐색"])
  self.session.youtube_search_scope="topic";self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  with redirect_stdout(io.StringIO()):self.tools["assess_current_screen"].invoke({"snapshot_id":1,"screen_kind":"results","context_correct":True,"target_visible":False,"evidence_text":"아이언맨","source":"ui"})
 def test_scroll_result_already_contains_fresh_image_without_another_model_observe_call(self):
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["아이언맨 다음 결과"]))
  model=InputModel(responses=[call("scroll_screen",{"direction":"down"},1),
   call("assess_youtube_screen",{"snapshot_id":2,"screen_kind":"player","query_correct":True,"target_visible":True,"evidence_text":"새 화면","source":"image"},2),
   call("finish_task",{"summary":"진행 중","goal_completed":False},3)])
  with patch("mobile_jarvis_app.agent.time.sleep"),redirect_stdout(io.StringIO()):build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":self.session.original_command}]})
  message=next(item for item in model.inputs[1] if isinstance(item,ToolMessage) and item.name=="scroll_screen")
  self.assertTrue(any(block.get("type")=="image_url" for block in message.content))
  self.assertEqual(self.device.read_screen.call_count,2)
  self.device.capture_screen.assert_called_once()
  self.assertEqual(self.session.performance["captures"],1)
  self.assertIn("assess_youtube_screen",model.available[1]);self.assertNotIn("observe_screen",model.available[1])
 def test_capture_failure_keeps_executed_action_and_observation_fallback(self):
  self.device.capture_screen.side_effect=ValueError("capture failed")
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["아이언맨 다음 결과"]))
  with patch("mobile_jarvis_app.agent.time.sleep"),redirect_stdout(io.StringIO()):self.tools["scroll_screen"].invoke({"direction":"down"})
  self.assertEqual(self.session.actions,1)
  self.assertEqual(self.session.action_history[-1]["status"],"executed")
  self.assertEqual(self.session.youtube_result_priority(),"observe")
 def test_other_app_does_not_gain_unrequested_capture_cost(self):
  self.session.screen=screen("other.app","목록")
  self.device.screen=self.session.screen
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",screen("other.app","다음 목록"))
  with patch("mobile_jarvis_app.agent.time.sleep"),redirect_stdout(io.StringIO()):self.tools["scroll_screen"].invoke({"direction":"down"})
  self.device.capture_screen.assert_not_called()

class ObservationCompactionTests(unittest.TestCase):
 def test_old_duplicate_context_removed_but_ui_labels_facts_latest_frames_preserved(self):
  payload={"screen":{"snapshot_id":1,"elements":[{"id":"1","label":"실제 제목","bounds":"[0,0][100,100]"}],"task_progress":{"observation_index":["x"*10000]},"workflow_guidance":{"a":"old"},"recent_actions":["old"]},"candidate":{"title":"저장한 제목"}}
  original=ToolMessage(content=json.dumps(payload),tool_call_id="1")
  messages=[original]+[ToolMessage(content=json.dumps({"i":i}),tool_call_id=str(i+2)) for i in range(4)]
  compact=compact_observation_history(messages)
  cleaned=json.loads(compact[0].content)
  self.assertNotIn("task_progress",cleaned["screen"])
  self.assertEqual(cleaned["screen"]["elements"],payload["screen"]["elements"])
  self.assertEqual(cleaned["candidate"],payload["candidate"])
  self.assertLess(len(compact[0].content),len(original.content)/5)
  self.assertIn("task_progress",json.loads(original.content)["screen"])
  self.assertEqual(compact[-1],messages[-1])
 def test_non_json_tool_errors_are_preserved(self):
  messages=[ToolMessage(content="도구 검증 오류",tool_call_id=str(i)) for i in range(6)]
  self.assertEqual([m.content for m in compact_observation_history(messages)],[m.content for m in messages])
