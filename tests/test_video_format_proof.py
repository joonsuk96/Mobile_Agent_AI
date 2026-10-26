"""Separate format proof from views, and detect contradictory Shorts UI structure."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession,build_tools,build_agent
from mobile_jarvis_app.device import parse_screen
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice,FakeCatalog,call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

TITLE="아이언맨 슈트 중 가장 까리한 슈트는??"
COMMAND="유튜브에서 아이언맨 쇼츠 말고 롱폼 조회수 100만 이상 재생해줘"
CARD=TITLE+" - 8분 10초 - 조회수 331만회"

def marked_screen(resource,full=False):
 bounds="[0,0][720,1500]" if full else "[0,100][720,500]"
 xml=f'<hierarchy><node package="{YOUTUBE_PACKAGE}" bounds="[0,0][720,1600]"><node package="{YOUTUBE_PACKAGE}" resource-id="{YOUTUBE_PACKAGE}:id/{resource}" bounds="{bounds}"><node package="{YOUTUBE_PACKAGE}" text="{CARD}" clickable="true" enabled="true" bounds="{bounds}"/></node></node></hierarchy>'
 return parse_screen(xml.encode(),{"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})

class FormatProofTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen([CARD,"일반 동영상 카드"])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND;self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("일반 영상 재생",[],["재생 확인"],["후보 확인","재생"])
  self.session.youtube_search_scope="topic";self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  with redirect_stdout(io.StringIO()):self.tools["assess_current_screen"].invoke({"snapshot_id":1,"screen_kind":"results","context_correct":True,"target_visible":True,"evidence_text":TITLE,"source":"ui"})
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def record(self,**extra):
  with redirect_stdout(io.StringIO()):return self.tools["record_video_candidate"].invoke({"snapshot_id":self.session.snapshot_id,"title":TITLE,"view_count_text":"조회수 331만회",**extra})
 def test_target_visible_alone_does_not_enable_candidate_click(self):
  self.assertEqual(self.session.youtube_result_priority(),"verify")
  with self.assertRaises(ValueError):self.tools["click_element"].invoke({"snapshot_id":1,"element_id":"0"})
  self.device.tap.assert_not_called()
 def test_views_only_candidate_is_deferred_and_releases_to_scroll(self):
  result=self.record()
  self.assertEqual(result["status"],"candidate_not_verified")
  self.assertIsNone(self.session.goal_tracker.candidate_selection)
  self.assertEqual(self.session.youtube_result_priority(),"scroll")
 def test_format_proof_allows_selection_but_short_ui_after_click_forces_back(self):
  result=self.record(content_kind="video",format_evidence="일반 동영상 카드")
  self.assertEqual(result["content_kind"],"video")
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",marked_screen("reel_watch_fragment_root",True))
  with redirect_stdout(io.StringIO()):self.tools["click_element"].invoke({"snapshot_id":1,"element_id":"0"})
  self.assertTrue(self.session.content_format_mismatch)
  self.assertIsNone(self.session.goal_tracker.candidate_selection)
  self.device.navigate.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([CARD,"일반 동영상 카드"]))
  model=ToolRecordingModel(responses=[call("navigate",{"direction":"back"},1),call("finish_task",{"summary":"탐색 중","goal_completed":False},2)])
  with redirect_stdout(io.StringIO()):build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":COMMAND}]})
  self.assertEqual(model.available[0],["navigate"])
  self.device.navigate.assert_called_once_with("back")
  result=self.record(content_kind="video",format_evidence="일반 동영상 카드")
  self.assertEqual(result["status"],"candidate_not_verified")
 def test_structural_short_card_cannot_be_labeled_as_regular_video(self):
  self.device.screen=marked_screen("shorts_shelf_item");self.session.observe()
  self.session.goal_tracker.remember(self.session.snapshot_id,self.device.screen,image_seen=True)
  result=self.record(content_kind="video",source="image",format_evidence="일반 동영상 카드")
  self.assertEqual(result["status"],"candidate_not_verified")
  self.assertIn("실제 UI 구조",result["reason"])
  self.assertIsNone(self.session.goal_tracker.candidate_selection)
 def test_short_player_detection_uses_viewport_not_inline_preview(self):
  self.assertTrue(marked_screen("reel_player_page",True)["shorts_player_visible"])
  self.assertFalse(marked_screen("reel_player_page",False)["shorts_player_visible"])
  self.assertEqual(marked_screen("shorts_shelf_item")["elements"][0]["content_type_hint"],"shorts")
 def test_duration_alone_is_not_format_proof(self):
  result=self.record(content_kind="video",format_evidence="8분 10초")
  self.assertEqual(result["status"],"candidate_not_verified")
 def test_normal_titles_do_not_create_structural_shorts_marker(self):
  page=youtube_screen(["쇼츠를 분석하는 일반 영상 - 조회수 331만회"])
  self.assertFalse(page["shorts_player_visible"])
