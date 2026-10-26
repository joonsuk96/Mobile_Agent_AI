"""Reduce model round trips without dropping content or stale-target validation."""
import io
import json
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock,patch
from langchain.messages import ToolMessage
from mobile_jarvis_app.agent import PhoneSession,build_agent,build_tools
from mobile_jarvis_app.context_window import compact_visual_history
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice,FakeCatalog,call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

TITLE="아이언맨 명장면"
CARD=TITLE+" - 8분 1초 - 조회수 254만회"
COMMAND="유튜브에서 아이언맨 쇼츠 말고 롱폼 조회수 100만 이상 재생해줘"

class FastSelectionTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen([CARD,"일반 동영상 카드","다른 정보 1"])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.device.read_screen=Mock(side_effect=lambda:self.device.screen)
  self.session=PhoneSession(self.device,FakeCatalog());self.session.original_command=COMMAND
  self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("일반 영상 재생",[],["목표 재생"],["후보 확인","재생"])
  self.session.youtube_search_scope="topic";self.session.observe()
  self.session.goal_tracker.remember(1,self.device.screen,image_seen=True)
  self.session.visual_snapshot={"snapshot_id":1,"captured_at":time.monotonic(),"image_url":"data:image/png;base64,eA==","width":100,"height":100,"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"}
  self.session.youtube_list_watch=True
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def args(self):
  return {"snapshot_id":1,"title":TITLE,"view_count_text":"조회수 254만회","source":"ui","content_kind":"video","format_evidence":"일반 동영상 카드","element_id":"0"}
 def test_one_model_tool_call_validates_and_clicks_without_separate_assess_record(self):
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([TITLE,"일시정지"]))
  model=ToolRecordingModel(responses=[call("select_video_candidate",self.args(),1),call("finish_task",{"summary":"탐색 중","goal_completed":False},2)])
  with redirect_stdout(io.StringIO()):build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":COMMAND}]})
  self.assertIn("select_video_candidate",model.available[0]);self.assertEqual(model.position,2)
  self.assertEqual(self.session.actions,1);self.device.tap.assert_called_once()
  self.assertEqual(self.device.read_screen.call_count,3)
  self.assertEqual(self.session.goal_tracker.candidate_selection["content_kind"],"video")
 def test_cached_ready_candidate_does_not_lose_snapshot_by_observing_again(self):
  with redirect_stdout(io.StringIO()):
   self.tools["record_video_candidate"].invoke({k:v for k,v in self.args().items() if k!="element_id"})
   self.tools["assess_current_screen"].invoke({"snapshot_id":1,"screen_kind":"results","context_correct":True,"target_visible":True,"evidence_text":TITLE,"source":"ui"})
   self.device.read_screen.reset_mock()
   self.tools["observe_screen"].func()
  self.device.read_screen.assert_not_called()
  self.assertEqual(self.session.snapshot_id,1)
 def test_unrelated_ui_change_does_not_force_reobservation_of_same_stable_target(self):
  self.device.screen=youtube_screen([CARD,"일반 동영상 카드","다른 정보 2"])
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([TITLE,"일시정지"]))
  with redirect_stdout(io.StringIO()):result,_=self.tools["select_video_candidate"].func(**self.args())
  self.device.tap.assert_called_once();self.assertIn("screen",result)
 def test_wrong_element_never_executes_click(self):
  args=self.args();args["element_id"]="2"
  with redirect_stdout(io.StringIO()):result,_=self.tools["select_video_candidate"].func(**args)
  self.assertEqual(result["status"],"not_selected");self.device.tap.assert_not_called()
 def test_missing_format_or_under_threshold_never_executes_click(self):
  args=self.args();args["content_kind"]="shorts"
  with redirect_stdout(io.StringIO()):result,_=self.tools["select_video_candidate"].func(**args)
  self.assertEqual(result["status"],"candidate_not_verified");self.device.tap.assert_not_called()

class ContextWindowTests(unittest.TestCase):
 def test_only_old_tool_images_are_removed_text_evidence_and_checkpoint_objects_unchanged(self):
  messages=[ToolMessage(content=[{"type":"text","text":str(i)},{"type":"image_url","image_url":{"url":str(i)}}],tool_call_id=str(i)) for i in range(5)]
  compact=compact_visual_history(messages)
  self.assertTrue(all(len(message.content)==1 for message in compact[:3]))
  self.assertEqual([message.content[0]["text"] for message in compact],[str(i) for i in range(5)])
  self.assertTrue(all(len(message.content)==2 for message in compact[3:]))
  self.assertTrue(all(len(message.content)==2 for message in messages))
 def test_plain_text_context_is_not_rewritten(self):
  messages=[ToolMessage(content="관찰 기록",tool_call_id="1")]
  self.assertIs(compact_visual_history(messages),messages)

class FastFirstTurnTests(unittest.TestCase):
 def test_explicit_first_app_command_skips_rewriting_model_call(self):
  script="""
import sys
from mobile_jarvis_app import agent
agent.build_model=lambda:(_ for _ in ()).throw(AssertionError('must skip preflight'))
agent.main=lambda:print('자비스 : '+sys.argv[2])
from mobile_jarvis_app.web_worker import run
run()
"""
  command="유튜브 앱 열어서 아이언맨 검색해줘"
  result=subprocess.run([sys.executable,"-u","-c",script],input=json.dumps({"command":command,"history":[],"conversation_enabled":True})+"\n",capture_output=True,text=True,encoding="utf-8",timeout=20)
  self.assertEqual(result.returncode,0,result.stderr)
  events=[json.loads(line) for line in result.stdout.splitlines()]
  self.assertIn({"type":"resolved","text":command},events)
