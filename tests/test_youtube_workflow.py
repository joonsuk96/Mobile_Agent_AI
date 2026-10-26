"""YouTube video-tab guidance and conditional scrolling evidence."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch
from xml.sax.saxutils import escape

from mobile_jarvis_app.agent import PhoneSession, build_tools, run_interactive
from mobile_jarvis_app.device import parse_screen
from mobile_jarvis_app.planning import GoalTracker, RankedCandidate
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE, youtube_video_request
from test_agent import FakeCatalog, FakeDevice, call
from test_app_launch_routing import ToolRecordingModel


def youtube_screen(labels, selected=False):
    nodes=[]
    for i,label in enumerate(labels):
        selected_attr = ' selected="true"' if selected and label in {"동영상", "Videos"} else ''
        nodes.append(f'<node package="{YOUTUBE_PACKAGE}" class="android.widget.TextView" enabled="true" clickable="true" text="{escape(label)}" bounds="[0,{i*30}][100,{i*30+25}]"{selected_attr}/>')
    return parse_screen(("<hierarchy>"+"".join(nodes)+"</hierarchy>").encode(),
                        {"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})


class YouTubeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.device=FakeDevice()
        self.device.screen=youtube_screen(["검색 결과", "동영상", "채널"])
        self.device.current_app=Mock(return_value={"package":YOUTUBE_PACKAGE,"activity":YOUTUBE_PACKAGE+"/.Main"})
        self.session=PhoneSession(self.device,FakeCatalog(),require_plan=False)
        self.session.original_command="유튜브에서 수찬 검색하고 3번째 최신 영상 재생"
        self.session.observe()
        self.tools={t.name:t for t in build_tools(self.session)}
        delay=patch("mobile_jarvis_app.agent.time.sleep");delay.start();self.addCleanup(delay.stop)

    def test_visible_video_tab_is_clicked_without_automatic_scroll(self):
        self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["동영상","최신순","첫 영상"],True))
        with redirect_stdout(io.StringIO()):
            result=self.tools["open_youtube_videos_tab"].invoke({"snapshot_id":1})
        self.assertTrue(result["opened"])
        self.device.tap.assert_called_once()
        self.device.scroll.assert_not_called()
        self.assertTrue(self.session.screen["elements"][0]["selected"])

    def test_selected_tab_is_not_clicked_or_scrolled_again(self):
        self.device.screen=youtube_screen(["동영상","최신순","요청 영상"],True)
        with redirect_stdout(io.StringIO()):
            result=self.tools["open_youtube_videos_tab"].invoke({"snapshot_id":1})
        self.assertTrue(result["already_selected"])
        self.device.tap.assert_not_called();self.device.scroll.assert_not_called()

    def test_missing_or_ambiguous_tab_is_not_guessed(self):
        for labels in [["채널","검색"],["동영상","동영상"]]:
            with self.subTest(labels=labels):
                self.device.screen=youtube_screen(labels)
                with redirect_stdout(io.StringIO()):
                    result=self.tools["open_youtube_videos_tab"].invoke({"snapshot_id":self.session.snapshot_id})
                self.assertFalse(result["opened"])
        self.device.tap.assert_not_called();self.device.scroll.assert_not_called()

    def test_search_submission_prioritizes_video_tab_tool(self):
        self.session.action_history=[{"action":"텍스트 입력","strategy":"type","status":"executed"},
                                     {"action":"클릭: 검색","strategy":"click","status":"executed"}]
        self.device.tap.side_effect=lambda *args:setattr(self.device,"screen",youtube_screen(["동영상","최신순","요청 영상"],True))
        model=ToolRecordingModel(responses=[call("open_youtube_videos_tab",{"snapshot_id":1},1),call("finish_task",{"summary":"동영상 목록을 열었어"},2)])
        with redirect_stdout(io.StringIO()):
            run_interactive(self.session,"유튜브에서 수찬 영상 찾아줘",model=model)
        self.assertEqual(model.available[0],["open_youtube_videos_tab"])
        self.device.tap.assert_called_once();self.device.scroll.assert_not_called()

    def test_search_box_opening_without_submission_does_not_force_tab(self):
        self.session.action_history=[{"action":"클릭: 검색","strategy":"click","status":"executed"}]
        self.assertFalse(self.session.youtube_videos_tab_due())

    def test_guide_is_not_applied_to_other_apps_or_explicit_other_content_types(self):
        self.assertTrue(youtube_video_request("유튜브 수찬 3번째 최신 영상 재생해줘"))
        for command in ["버스타고에서 버스 찾아줘","유튜브 앱 켜줘","유튜브 쇼츠 영상 재생","유튜브 플레이리스트 틀어줘"]:
            with self.subTest(command=command):self.assertFalse(youtube_video_request(command))

    def test_ranked_candidates_can_span_before_and_after_scroll(self):
        tracker=GoalTracker("3번째 최신 영상 재생")
        tracker.set_plan("세 번째 영상",["최신순"],["요청 영상 재생"],["정렬 확인","후보 확인"])
        tracker.remember(1,youtube_screen(["최신순","첫 영상","1일 전","둘째 영상","2일 전"]))
        tracker.remember(2,youtube_screen(["둘째 영상","2일 전","셋째 영상","3일 전"]))
        candidates=[RankedCandidate(position=1,title="첫 영상",publication_text="1일 전",snapshot_id=1),
                    RankedCandidate(position=2,title="둘째 영상",publication_text="2일 전",snapshot_id=1),
                    RankedCandidate(position=3,title="셋째 영상",publication_text="3일 전",snapshot_id=2)]
        result=tracker.record_ranked(1,"최신순",candidates,"ui","newest",True)
        self.assertEqual(result["title"],"셋째 영상")
        candidates[2].snapshot_id=1
        with self.assertRaises(ValueError):tracker.record_ranked(1,"최신순",candidates,"ui","newest",True)
