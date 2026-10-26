"""YouTube result lists prefer selecting a visible target or scrolling if absent."""
import base64
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_agent, build_tools
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice, FakeCatalog, call, screen
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen
from test_browser_fallback import PNG

COMMAND="유튜브에서 아이언맨 쇼츠 말고 긴 동영상 조회수 100만 이상 재생해줘"
HIGH="아이언맨 명장면 - 8분 10초 - 조회수 331만회 - 동영상 재생"

class ScrollPriorityTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.device.screen=youtube_screen(["아이언맨 검색 결과",HIGH])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.device.capture_screen=Mock(return_value={"image_url":"data:image/png;base64,"+base64.b64encode(PNG).decode(),"width":1,"height":1,"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND;self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("조건 영상 재생",[],["조건 확인"],["검색","선택"])
  self.session.youtube_search_scope="topic";self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def assess(self,visible,kind="results",correct=True):
  with redirect_stdout(io.StringIO()):self.tools["assess_current_screen"].invoke({"snapshot_id":self.session.snapshot_id,"screen_kind":kind,"context_correct":correct,"target_visible":visible,"evidence_text":"아이언맨","source":"ui"})
 def invoke(self,responses):
  model=ToolRecordingModel(responses=responses)
  with redirect_stdout(io.StringIO()) as out:build_agent(self.session,model=model).invoke({"messages":[{"role":"user","content":COMMAND}]})
  return model,out.getvalue()
 def test_absent_target_only_offers_scroll_or_panel_swipe(self):
  self.assess(False)
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["아이언맨",HIGH]))
  model,_=self.invoke([
   call("scroll_screen",{"direction":"down"},1),
   call("assess_youtube_screen",{"snapshot_id":2,"screen_kind":"player","query_correct":True,"target_visible":True,"evidence_text":"플레이어","source":"image"},3),
   call("finish_task",{"summary":"탐색 중","goal_completed":False},4)])
  self.assertEqual(set(model.available[0]),{"scroll_screen","swipe_region"})
  self.assertIn("assess_youtube_screen",model.available[1]);self.assertNotIn("observe_screen",model.available[1])
  self.device.scroll.assert_called_once_with("down")
  self.device.capture_screen.assert_called_once()
  self.device.navigate.assert_not_called()
 def test_absent_target_blocks_filter_click_before_it_reaches_phone(self):
  self.assess(False)
  with self.assertRaisesRegex(ValueError,"스크롤"):
   self.session.perform("클릭: 옵션 더보기",lambda:self.device.tap(1,1))
  self.device.tap.assert_not_called()
 def test_visible_target_blocks_scroll_and_allows_click(self):
  self.assess(True)
  self.session.goal_tracker.remember(1,self.device.screen,image_seen=True)
  with redirect_stdout(io.StringIO()):self.tools["record_video_candidate"].invoke({"snapshot_id":1,"title":"아이언맨 명장면","view_count_text":"조회수 331만회","source":"image","content_kind":"video","format_evidence":"일반 동영상 카드 구조를 확인"})
  with self.assertRaisesRegex(ValueError,"목표가 보인다고"):
   self.tools["scroll_screen"].invoke({"direction":"down"})
  self.device.scroll.assert_not_called()
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["아이언맨 명장면","일시정지"]))
  model,_=self.invoke([call("click_element",{"snapshot_id":1,"element_id":"1"},1),call("finish_task",{"summary":"탐색 중","goal_completed":False},2)])
  self.assertIn("click_element",model.available[0]);self.assertNotIn("scroll_screen",model.available[0]);self.assertNotIn("navigate",model.available[0])
  self.device.tap.assert_called_once();self.assertIsNone(self.session.youtube_result_priority())
 def test_non_list_wrong_context_player_and_other_apps_are_not_forced_to_scroll(self):
  for kind,correct in [("dialog",True),("search_editor",True),("player",True),("results",False)]:
   self.assess(False,kind,correct);self.assertIsNone(self.session.youtube_result_priority())
  self.assess(False)
  self.session.screen=screen("other.package","아이언맨")
  self.assertIsNone(self.session.youtube_result_priority())
 def test_stalled_scroll_releases_priority_but_does_not_complete(self):
  self.assess(False);self.session.youtube_result_scrolls=3
  self.assertEqual(self.session.youtube_result_priority(),"stalled")
  model,_=self.invoke([call("finish_task",{"summary":"탐색 중","goal_completed":False},1)])
  self.assertIn("click_element",model.available[0]);self.assertFalse(self.session.completed)
 def test_channel_entry_and_finalization_keep_their_existing_routes(self):
  self.assess(False);self.session.youtube_search_scope="channel";self.session.youtube_channel_pending=True
  self.assertIsNone(self.session.youtube_result_priority())
  self.session.youtube_channel_pending=False;self.session.finalization_pending=True
  self.assertIsNone(self.session.youtube_result_priority())
 def test_image_candidate_record_resets_false_no_new_scroll_counter(self):
  memory=self.session.youtube_search_memory;memory.no_new_scrolls=2
  from mobile_jarvis_app.workflows import SearchVideoCard
  memory.record_page("아이언맨",[SearchVideoCard(title="새 항목",view_count_text="조회수 90만회")],2)
  self.assertEqual(memory.no_new_scrolls,0)
