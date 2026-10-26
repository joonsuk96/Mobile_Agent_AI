"""General-topic search must stay on results and explore additional candidates."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.media import view_requirement, displayed_views, meets_views
from mobile_jarvis_app.workflows import YouTubeSearchMemory, youtube_context, YOUTUBE_PACKAGE
from test_agent import FakeDevice, FakeCatalog, call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

COMMAND="유튜브에서 아이언맨 검색하고 조회수 천만 넘는 영상 아무거나 재생해줘"
LOW="아이언맨 예고편 - 3분 1초 - 조회수 90만회 - 동영상 재생"
HIGH="아이언맨 명장면 모음 - 8분 1초 - 조회수 1200만회 - 동영상 재생"
TITLE="아이언맨 명장면 모음"

class NaturalViewCountsTests(unittest.TestCase):
 def test_korean_numbers_and_strict_comparison(self):
  requirement=view_requirement(COMMAND)
  self.assertEqual(requirement,{"value":10000000,"operator":"초과"})
  self.assertFalse(meets_views(10000000,requirement))
  self.assertTrue(meets_views(12000000,requirement))
  for command,value in [("조회수 백만 이상",1000000),("조회수 1천만 넘는",10000000),("조회수가 50만 보다 많은",500000),("조회수 십만 이하",100000)]:
   with self.subTest(command=command):self.assertEqual(view_requirement(command)["value"],value)
  for text,value in [("조회수 1.2천회",1200),("조회수 천만회",10000000),("조회수 50만회",500000)]:
   with self.subTest(text=text):self.assertEqual(displayed_views(text)["value"],value)

class TopicSearchTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice()
  self.device.screen=youtube_screen(["검색 결과","동영상","채널 보기",LOW])
  self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":".Main"})
  self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND
  self.session.goal_tracker.original_command=COMMAND
  self.session.goal_tracker.set_plan("아이언맨 조건 영상 재생",["천만 초과"],["영상 조회수 조건 확인","선택 영상 재생"],["결과 탐색","조건 확인","재생"])
  self.session.youtube_search_scope="topic"
  self.session.youtube_search_memory.submit("아이언맨")
  self.session.observe()
  self.tools={tool.name:tool for tool in build_tools(self.session)}
  delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)
 def test_topic_does_not_force_channel_or_channel_tab(self):
  self.session.youtube_channel_pending=True
  self.session.youtube_videos_pending=True
  self.assertFalse(self.session.youtube_channel_due())
  self.assertFalse(self.session.youtube_videos_tab_due())
  guidance=self.session.youtube_guidance()
  self.assertEqual(guidance["search_scope"],"topic")
  self.assertIn("검색 결과의 영상 목록",guidance["next_step_policy"])
  self.assertFalse(guidance["search_memory"]["visible_candidates"][0]["views_match"])
 def test_wrong_channel_plan_can_switch_to_topic_without_changing_goal(self):
  self.session.youtube_search_scope="channel"
  self.session.youtube_channel_pending=True
  before=self.session.goal_tracker.state()["criteria"]
  self.tools["set_youtube_search_strategy"].invoke({"scope":"topic","reason":"아이언맨은 주제이며 요청 채널이 아니다"})
  self.assertFalse(self.session.youtube_channel_pending)
  self.assertEqual(self.session.goal_tracker.criteria,before)
  self.assertEqual(self.session.actions,0)
 def test_scroll_reveals_matching_video_then_completes_without_back_or_research(self):
  self.device.scroll.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["검색 결과",HIGH]))
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen([TITLE,"일시정지"]))
  model=ToolRecordingModel(responses=[
   call("scroll_screen",{"direction":"down"},1),
   call("click_element",{"snapshot_id":2,"element_id":"1"},2),
   call("finish_task",{"summary":"천만 조회수를 넘는 아이언맨 명장면 모음을 재생했어.","completion_checks":[
    {"criterion_id":"C1","snapshot_id":2,"evidence_text":"조회수 1200만회","source":"ui"},
    {"criterion_id":"C2","snapshot_id":3,"evidence_text":"일시정지","source":"ui"}],
    "selected_item":{"snapshot_id":3,"visible_title":TITLE,"source":"ui"}},3)])
  with redirect_stdout(io.StringIO()):run_interactive(self.session,COMMAND,model=model)
  self.assertTrue(self.session.completed)
  self.device.scroll.assert_called_once_with("down")
  self.device.navigate.assert_not_called()
  self.assertIn("scroll_screen",model.available[0])
  self.assertEqual(self.session.goal_tracker.candidate_selection["view_count"],12000000)
 def test_auto_scope_does_not_assume_a_channel_after_search(self):
  self.session.youtube_search_scope="auto"
  self.session.youtube_query="아이언맨"
  self.session.youtube_query_pending=True
  self.session.action_history.append({"action":"텍스트 입력","strategy":"type","status":"executed"})
  self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["검색 결과",LOW]))
  with redirect_stdout(io.StringIO()):self.session.perform("클릭: 검색",lambda:self.device.tap(10,10))
  self.assertFalse(self.session.youtube_channel_pending)
  self.assertEqual(self.session.youtube_search_memory.searches,["아이언맨","아이언맨"])

class SearchMemoryTests(unittest.TestCase):
 def test_seen_cards_and_stalled_scrolls_persist_across_observations(self):
  memory=YouTubeSearchMemory();memory.submit("아이언맨")
  page=youtube_screen([LOW])
  memory.observe(page,1,"클릭: 검색")
  memory.observe(page,2,"스크롤: down")
  memory.observe(page,2,"스크롤: down")
  self.assertEqual(memory.no_new_scrolls,1)
  memory.observe(page,3,"영역 스와이프")
  state=memory.state(COMMAND)
  self.assertEqual(state["candidates_examined_for_query"],1)
  self.assertEqual(state["scrolls_without_new_candidates"],2)
  self.assertIn("필터/검색어",state["decision_hint"])
  memory.observe(youtube_screen([HIGH]),4,"스크롤: down")
  self.assertEqual(memory.no_new_scrolls,0)
  self.assertTrue(memory.state(COMMAND)["visible_candidates"][0]["views_match"])
