import unittest
from datetime import date
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock
from mobile_jarvis_app.comparison import comparison_requirement, PeriodComparisonEvidence, validate_period_comparison
from mobile_jarvis_app.planning import GoalTracker
from mobile_jarvis_app.agent import PhoneSession, build_tools
from test_agent import FakeDevice, FakeCatalog, screen

COMMAND = '유튜브 열어서 황룡갑 검색해서 최근 3개월 내 올라온 영상중에 가장 조회수 높은 영상 제목이 무엇이며 조회수가 얼마야'

class PeriodComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tracker = GoalTracker(COMMAND)
        self.tracker.remember(1, screen(text='최신순'))
        self.items = [dict(title='영상 A', views_text='조회수 12만회', publication_text='2026-10-01', snapshot_id=2),
                      dict(title='영상 B', views_text='조회수 30만회', publication_text='2026-08-01', snapshot_id=3),
                      dict(title='기간 밖 영상', views_text='조회수 90만회', publication_text='2026-07-01', snapshot_id=4)]
        for item in self.items:
            self.tracker.remember(item['snapshot_id'], screen(text=' '.join([item['title'], item['views_text'], item['publication_text']])))
        self.proof = dict(sort_snapshot_id=1, sort_evidence='최신순', started_at_newest=True,
                          continuous_coverage=True, all_candidates_recorded=True,
                          candidates=self.items, selected_title='영상 B')
    def validate(self, proof=None, summary='영상 B 조회수 30만회'):
        validate_period_comparison(self.tracker, PeriodComparisonEvidence(**(proof or self.proof)), summary, date(2026,10,26))
    def test_complete_range_selects_maximum_not_outside_video(self):
        self.validate()
    def test_one_or_two_screens_without_boundary_cannot_finish(self):
        with self.assertRaisesRegex(ValueError, '경계'):
            self.validate(dict(self.proof, candidates=self.items[:2]))
    def test_missing_proof_cannot_finish(self):
        with self.assertRaises(ValueError):
            validate_period_comparison(self.tracker, None, '', date(2026,10,26))
    def test_wrong_winner_rejected(self):
        with self.assertRaisesRegex(ValueError, '최대 조회수'):
            self.validate(dict(self.proof, selected_title='영상 A'))
    def test_coverage_gap_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(dict(self.proof, continuous_coverage=False))
    def test_invented_ui_evidence_rejected(self):
        items=[dict(self.items[0], views_text='조회수 999만회'), *self.items[1:]]
        with self.assertRaises(ValueError):
            self.validate(dict(self.proof, candidates=items))
    def test_ambiguous_three_month_boundary_needs_exact_date(self):
        self.tracker.remember(4, screen(text='기간 밖 영상 조회수 90만회 3개월 전'))
        items=[*self.items[:2],dict(self.items[2],publication_text='3개월 전')]
        with self.assertRaisesRegex(ValueError,'모호'):
            self.validate(dict(self.proof,candidates=items))
    def test_short_channel_can_finish_with_actual_list_end(self):
        self.tracker.remember(5, screen(text='목록 끝'))
        self.validate(dict(self.proof,candidates=self.items[:2],list_end_snapshot_id=5,list_end_evidence='목록 끝'))
    def test_ordinary_commands_are_unaffected(self):
        for command in ('유튜브 켜줘','조회수 100만 이상 영상 재생','두 번째 최신 영상','고구마스틱 가격 알려줘'):
            self.assertIsNone(comparison_requirement(command))
        validate_period_comparison(GoalTracker('유튜브 켜줘'),None,'')
    def test_comparison_observations_survive_long_search(self):
        for index in range(5,40):
            self.tracker.remember(index,screen(text='다음 페이지'))
        self.assertIn(1,self.tracker.observations)
        ordinary=GoalTracker('앱 열기')
        for index in range(40): ordinary.remember(index,screen())
        self.assertEqual(len(ordinary.observations),30)
    def test_premature_finish_resumes_actions_without_device_or_model_call(self):
        session=PhoneSession(FakeDevice(),FakeCatalog())
        session.original_command=COMMAND
        session.goal_tracker.original_command=COMMAND
        session.device.read_screen=Mock(side_effect=AssertionError('no extra read'))
        session.final_response_model=Mock()
        tools={item.name:item for item in build_tools(session)}
        with redirect_stdout(StringIO()):
            result=tools['finish_task'].invoke(dict(summary='영상 A 조회수 12만회',goal_completed=True,completion_checks=[]))
        self.assertEqual(result['status'],'not_verified')
        self.assertEqual(result['phase'],'actions')
        self.assertFalse(session.finalization_pending)
        self.assertEqual(session.completion_validation_failures,0)
        session.device.read_screen.assert_not_called()
        session.final_response_model.invoke.assert_not_called()
    def test_comparison_schema_is_absent_from_unrelated_commands(self):
        session=PhoneSession(FakeDevice(),FakeCatalog())
        session.original_command='유튜브 켜줘'
        tools={item.name:item for item in build_tools(session)}
        self.assertNotIn('period_comparison',tools['finish_task'].args_schema.model_fields)
    def test_valid_comparison_finishes_with_only_existing_final_device_read(self):
        from unittest.mock import patch
        session=PhoneSession(FakeDevice(),FakeCatalog())
        session.original_command=COMMAND
        session.goal_tracker=self.tracker
        session.goal_tracker.set_plan('기간 내 최다 조회수 확인',[],['결과 확인'],['비교'])
        session.screen=screen(text='영상 B 조회수 30만회')
        session.device.screen=session.screen
        session.device.read_screen=Mock(return_value=session.screen)
        session.final_response_model=Mock()
        tools={item.name:item for item in build_tools(session)}
        with patch('mobile_jarvis_app.comparison.datetime') as clock, redirect_stdout(StringIO()):
            clock.now.return_value.date.return_value=date(2026,10,26)
            result=tools['finish_task'].invoke(dict(summary='영상 B 조회수 30만회',goal_completed=True,
                completion_checks=[dict(criterion_id='C1',snapshot_id=3,source='ui',evidence_text='영상 B')],
                period_comparison=self.proof))
        self.assertEqual(result['status'],'completed')
        session.device.read_screen.assert_called_once()
        session.final_response_model.invoke.assert_not_called()
    def test_comparison_target_visible_does_not_force_single_selection(self):
        session=PhoneSession(FakeDevice(),FakeCatalog())
        session.original_command=COMMAND
        session.screen=screen()
        session.current_screen_assessment=Mock(return_value=dict(screen_kind='results',context_correct=True,target_visible=True))
        self.assertIsNone(session.youtube_result_priority())

if __name__=='__main__': unittest.main()
