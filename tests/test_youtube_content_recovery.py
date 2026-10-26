"""Common result exploration must not exclude playlists, music, Shorts or live."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.workflows import youtube_content_request, youtube_video_request, youtube_context, YOUTUBE_PACKAGE
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

COMMAND="유튜브로 내가 좋아하는 음악 플레이리스트 아무거나 조회수 높아보이는거 틀어줘"

class ContentRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen(["음악 플레이리스트","검색"])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND;self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("플레이리스트 재생",[],["콘텐츠 재생"],["검색","후보 탐색"])
  self.session.youtube_search_scope="channel"  # Even a mistaken scope must not force Videos.
  self.session.youtube_channel_pending=True;self.session.youtube_videos_pending=True
  self.session.action_history=[{"action":name,"status":"executed","strategy":"click"} for name in ["클릭: 검색","이동: back","클릭: 검색","이동: back"]]
  self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def test_all_content_types_receive_common_guide_without_enabling_channel_video_route(self):
  for command in [COMMAND,"유튜브에서 음악 틀어줘","유튜브 재생목록 찾아줘","유튜브 쇼츠 틀어줘","유튜브 라이브 재생해줘"]:
   with self.subTest(command=command):
    self.assertTrue(youtube_content_request(command));self.assertFalse(youtube_video_request(command))
    self.assertIsNotNone(youtube_context(command,self.device.screen))
  self.assertFalse(youtube_content_request("유튜브 앱 열어줘"))
  self.assertFalse(youtube_content_request("다른 앱에서 음악 틀어줘"))
 def test_playlist_context_and_loop_detection_are_active(self):
  context=self.session.youtube_guidance()
  self.assertEqual(context["content_kind"],"playlist")
  self.assertIn("검색 결과의 영상 목록",context["next_step_policy"])
  self.assertTrue(self.session.youtube_cycle_state()["recovery_due"])
  self.assertFalse(self.session.youtube_channel_due());self.assertFalse(self.session.youtube_videos_tab_due())
 def test_playlist_cycle_requires_screen_assessment_then_allows_model_chosen_scroll(self):
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["다른 플레이리스트","음악 모음"]))
  model=ToolRecordingModel(responses=[
   call("assess_youtube_screen",{"snapshot_id":1,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"음악 플레이리스트","source":"ui"},1),
   call("scroll_screen",{"direction":"down"},2),
   call("finish_task",{"summary":"탐색 중","goal_completed":False},3)])
  with redirect_stdout(io.StringIO()):build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":COMMAND}]})
  self.assertEqual(model.available[0],["assess_youtube_screen"])
  self.assertIn("scroll_screen",model.available[1]);self.device.scroll.assert_called_once_with("down")
  self.device.navigate.assert_not_called()
 def test_mistaken_channel_scope_does_not_block_playlist_candidate_card(self):
  self.device.screen=youtube_screen(["편안한 음악 플레이리스트 - 60분 1초 - 조회수 200만회"])
  self.session.observe()
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["편안한 음악 플레이리스트","일시정지"]))
  with redirect_stdout(io.StringIO()):self.tools["click_element"].invoke({"snapshot_id":2,"element_id":"0"})
  self.device.tap.assert_called_once()
 def test_assessed_playlist_results_cannot_be_immediately_discarded_with_back(self):
  with redirect_stdout(io.StringIO()):self.tools["assess_youtube_screen"].invoke({"snapshot_id":1,"screen_kind":"results","query_correct":True,"target_visible":False,"evidence_text":"음악 플레이리스트","source":"ui"})
  with self.assertRaisesRegex(ValueError,"결과 목록"):self.tools["navigate"].invoke({"direction":"back"})
  self.device.navigate.assert_not_called()
