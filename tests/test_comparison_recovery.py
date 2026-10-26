import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock
from mobile_jarvis_app.agent import PhoneSession, build_tools
from mobile_jarvis_app.comparison import ComparisonEvidenceError, ComparisonSortMissing, check_recorded_evidence
from mobile_jarvis_app.planning import GoalTracker
from test_agent import FakeDevice, FakeCatalog, screen
from test_period_comparison import COMMAND

class ComparisonRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.session=PhoneSession(FakeDevice(),FakeCatalog())
        self.session.original_command=COMMAND
        self.session.goal_tracker.original_command=COMMAND
        self.session.goal_tracker.set_plan('비교',[],['비교 완료'],['탐색'])
        self.session.screen=screen(text='조회 결과')
        self.session.device.read_screen=Mock(side_effect=AssertionError('unexpected read'))
        self.tools={t.name:t for t in build_tools(self.session)}
    def invoke(self,**kwargs):
        with redirect_stdout(StringIO()):
            return self.tools['finish_task'].invoke(dict(summary='영상 A 70만회, 최신순 정렬 근거 부족',goal_completed=False,remaining_work='최신순 정렬 확인',completion_checks=[])|kwargs)
    def test_false_completion_preserves_list_and_blocks_back_before_io(self):
        result=self.invoke()
        self.assertEqual(result['phase'],'comparison_inspect_sort')
        with self.assertRaises(ValueError):
            self.session.perform('이동: back',Mock())
        self.session.device.read_screen.assert_not_called()
        self.assertEqual(self.session.actions,0)
    def test_reentering_channel_or_video_tab_is_blocked(self):
        self.invoke()
        for name in ('클릭: 채널로 이동','클릭: 동영상'):
            with self.assertRaises(ValueError): self.session.perform(name,Mock())
    def test_actual_missing_range_remains_discovery_not_metadata_failure(self):
        result=self.invoke(summary='아직 경계 확인 전',remaining_work='다음 페이지 비교')
        self.assertEqual(result['phase'],'comparison_coverage')
        self.assertFalse(self.session.finalization_pending)
        self.assertEqual(self.session.completion_validation_failures,0)
    def test_exact_old_ui_witness_repairs_wrong_snapshot_without_io(self):
        tracker=self.session.goal_tracker
        tracker.remember(1,screen(text='최신순'))
        tracker.remember(8,screen(text='다른 목록'))
        check_recorded_evidence(tracker,8,'ui',['최신순'],sort=True)
        self.session.device.read_screen.assert_not_called()
    def test_missing_sort_is_targeted_inspection_not_full_discovery(self):
        tracker=self.session.goal_tracker
        tracker.remember(8,screen(text='다른 목록'))
        with self.assertRaises(ComparisonSortMissing):
            check_recorded_evidence(tracker,8,'ui',['최신순'],sort=True)
    def test_fabricated_candidate_is_metadata_error_not_coverage(self):
        tracker=self.session.goal_tracker
        tracker.remember(8,screen(text='영상 A 조회수 70만회 2주 전'))
        with self.assertRaises(ComparisonEvidenceError):
            check_recorded_evidence(tracker,8,'ui',['영상 A','조회수 700만회','2주 전'])
    def test_candidate_fields_cannot_be_assembled_from_different_frames(self):
        tracker=self.session.goal_tracker
        tracker.remember(1,screen(text='영상 A 조회수 70만회'))
        tracker.remember(2,screen(text='다른 영상 2주 전'))
        with self.assertRaises(ComparisonEvidenceError):
            check_recorded_evidence(tracker,1,'ui',['영상 A','조회수 70만회','2주 전'])
    def test_wrong_app_witness_does_not_repair_sort(self):
        tracker=self.session.goal_tracker
        tracker.remember(1,screen('other.app','최신순'))
        tracker.remember(2,screen(text='다른 목록'))
        with self.assertRaises(ComparisonSortMissing):
            check_recorded_evidence(tracker,2,'ui',['최신순'],sort=True)
    def test_escape_needs_actual_wrong_path_assessment(self):
        self.invoke()
        self.session.current_screen_assessment=Mock(return_value={'context_correct':True})
        with self.assertRaises(ValueError):
            self.tools['resume_task_actions'].invoke(dict(criterion_id='comparison_path',reason='sort',why_observations_insufficient='sort'))
        self.session.current_screen_assessment=Mock(return_value={'context_correct':False})
        result=self.tools['resume_task_actions'].invoke(dict(criterion_id='comparison_path',reason='잘못된 화면',why_observations_insufficient='실제 다른 경로'))
        self.assertEqual(result['status'],'actions_resumed')
        self.assertIsNone(self.session.comparison_recovery)
    def test_noncomparison_finish_behavior_unchanged(self):
        self.session.original_command='유튜브 켜줘'
        self.invoke()
        self.assertIsNone(self.session.comparison_recovery)

if __name__=='__main__':unittest.main()
