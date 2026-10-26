"""Source correction uses existing observations, never more model or device calls."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import Mock,patch
from mobile_jarvis_app.agent import PhoneSession,run_interactive
from mobile_jarvis_app.planning import SelectedItemEvidence
from mobile_jarvis_app.workflows import YOUTUBE_PACKAGE
from test_agent import FakeDevice,FakeCatalog,call
from test_app_launch_routing import ToolRecordingModel
from test_youtube_workflow import youtube_screen

TITLE="존박&곽진언 랩에 푹 빠진 로꼬 #shorts"
COMMAND="유튜브에서 존박 랩 검색하고 쇼츠 조회수 200만 이상 아무거나 틀어줘"

class ZeroIOCompletionTests(unittest.TestCase):
 def setUp(self):
  self.device=FakeDevice();self.session=PhoneSession(self.device,FakeCatalog())
  self.session.original_command=COMMAND;tracker=self.session.goal_tracker;tracker.original_command=COMMAND
  tracker.set_plan("조건 쇼츠 재생",[],["YouTube 앱이 실행됨","조회수 200만 이상","선택 쇼츠가 재생 중"],["검색","선택","재생 확인"])
  tracker.app_names[YOUTUBE_PACKAGE]={"YouTube","유튜브"}
  tracker.remember(1,youtube_screen(["검색"]))
  tracker.remember(2,youtube_screen([]),image_seen=True)
  tracker.register_candidate(2,TITLE,"조회수 211만회","image")
  self.device.screen=youtube_screen([TITLE,"일시중지"])
  self.session.observe()  # snapshot is reset below to model the post-click frame.
  self.session.snapshot_id=3;self.session.screen=self.device.screen
  tracker.remember(3,self.device.screen)
  self.device.read_screen=Mock(return_value=self.device.screen)
  self.device.capture_screen=Mock(side_effect=AssertionError("no extra capture"))
 def test_current_ui_title_and_playback_correct_image_source_without_capture(self):
  tracker=self.session.goal_tracker
  title=tracker.evidence(3,"image",TITLE)
  self.assertEqual(title["source"],"ui")
  self.assertEqual(title["resolution"],"same_frame_ui")
  self.assertTrue(tracker.confirm_selected(3,TITLE,"image")["matches"])
  self.device.read_screen.assert_not_called();self.device.capture_screen.assert_not_called()
 def test_single_finish_call_completes_without_retry_model_or_extra_capture(self):
  model=ToolRecordingModel(responses=[call("finish_task",{
   "summary":f"조회수 211만 회인 쇼츠 ‘{TITLE}’를 재생했어.",
   "completion_checks":[
    {"criterion_id":"C1","snapshot_id":1,"source":"app","evidence_text":"YouTube 앱이 전면에서 실행 중임을 확인"},
    {"criterion_id":"C2","snapshot_id":2,"source":"image","evidence_text":"조회수 211만회"},
    {"criterion_id":"C3","snapshot_id":3,"source":"image","evidence_text":"일시중지"}],
   "selected_item":{"snapshot_id":3,"visible_title":TITLE,"source":"image"}},1)])
  with redirect_stdout(io.StringIO()) as output:run_interactive(self.session,COMMAND,model=model)
  self.assertTrue(self.session.completed)
  self.assertEqual(model.position,1)
  self.assertEqual(self.session.completion_validation_failures,0)
  self.assertEqual(self.device.read_screen.call_count,1) # existing final foreground check only
  self.device.capture_screen.assert_not_called()
  self.assertIn("[완료 확정]",output.getvalue())
  self.assertNotIn("완료 근거 보완",output.getvalue())
 def test_absent_title_and_absent_image_remain_unverified(self):
  with self.assertRaises(ValueError):self.session.goal_tracker.evidence(3,"image","다른 영상 제목")
  self.device.capture_screen.assert_not_called()
 def test_embedded_real_package_resolves_but_wrong_package_does_not(self):
  tracker=self.session.goal_tracker
  evidence=tracker.resolve_record_evidence("C1","package: com.google.android.youtube",1,"app",[])
  self.assertEqual(evidence["text"],YOUTUBE_PACKAGE)
  for text in ("package: com.google.android.youtube.music","fake.app.package"):
   with self.subTest(text=text),self.assertRaises(ValueError):tracker.resolve_record_evidence("C1",text,1,"app",[])
