"""Preserve verified card facts and tolerate only trailing format-tag display changes."""
import unittest
from mobile_jarvis_app.planning import GoalTracker,CompletionCheck,SelectedItemEvidence,comparable_title
from test_youtube_workflow import youtube_screen

TITLE="존박&곽진언 랩에 푹 빠진 로꼬 #shorts"
CORE="존박&곽진언 랩에 푹 빠진 로꼬"

class TitleProvenanceTests(unittest.TestCase):
 def setUp(self):
  self.tracker=GoalTracker("유튜브에서 존박 랩 쇼츠 조회수 200만 이상 틀어줘")
  self.tracker.set_plan("조건 쇼츠 재생",[],["유튜브 실행","조회수 200만 이상","쇼츠 형식","선택 영상 재생"],["후보","선택"])
  self.tracker.app_names["com.google.android.youtube"]={"YouTube","유튜브"}
  self.tracker.remember(1,youtube_screen(["검색"]))
  self.tracker.remember(2,youtube_screen([]),image_seen=True)
  candidate=self.tracker.register_candidate(2,TITLE,"조회수 212만회","image")
  candidate.update(content_kind="shorts",format_evidence="Shorts 영역 카드")
  self.tracker.remember(3,youtube_screen([CORE,"일시중지"]))
 def test_old_selected_snapshot_rebases_to_literal_current_title_without_format_tag(self):
  result=self.tracker.confirm_selected(2,TITLE,"image")
  self.assertTrue(result["matches"])
  self.assertEqual(result["snapshot_id"],3)
  self.assertEqual(self.tracker.candidate_selection["view_count"],2120000)
  self.assertEqual(self.tracker.candidate_selection["content_kind"],"shorts")
 def test_existing_format_views_and_current_pause_resolve_completion_without_more_observation(self):
  checks=[CompletionCheck(criterion_id="C1",snapshot_id=1,source="app",evidence_text="com.google.android.youtube"),
   CompletionCheck(criterion_id="C2",snapshot_id=3,source="ui",evidence_text="조회수 212만회"),
   CompletionCheck(criterion_id="C3",snapshot_id=3,source="ui",evidence_text="쇼츠"),
   CompletionCheck(criterion_id="C4",snapshot_id=3,source="ui",evidence_text="선택한 쇼츠가 재생 중")]
  missing=self.tracker.apply_completion(checks,None,SelectedItemEvidence(snapshot_id=2,visible_title=TITLE,source="image"),{},[])
  self.assertEqual(missing,[])
  self.assertEqual(self.tracker.criteria["C4"]["evidence"]["text"],"일시중지")
 def test_different_title_is_never_matched_by_a_short_prefix(self):
  self.tracker.remember(4,youtube_screen(["존박&곽진언 다른 공연","일시중지"]))
  with self.assertRaises(ValueError):self.tracker.confirm_selected(2,TITLE,"image")
  self.assertFalse(self.tracker.selected_item_matches)
 def test_nonformat_hashtags_remain_part_of_identity(self):
  self.assertNotEqual(comparable_title("공연 #episode1"),comparable_title("공연 #episode2"))
 def test_format_fact_alone_does_not_verify_playback(self):
  self.tracker.criteria["C4"]["description"]="쇼츠 재생 확인"
  self.tracker.remember(4,youtube_screen(["검색창"]))
  with self.assertRaises(ValueError):self.tracker.record("C4","satisfied","쇼츠",4,"ui",{})
