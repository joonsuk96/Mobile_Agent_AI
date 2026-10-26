"""Negated content mentions must not select the excluded content route."""
import unittest
from mobile_jarvis_app.workflows import youtube_content_kind, youtube_video_request, youtube_format_requirement, youtube_context
from mobile_jarvis_app.media import view_requirement

class ContentExclusionTests(unittest.TestCase):
 def test_excluded_shorts_are_regular_video_requests(self):
  for wording in ["쇼츠 말고 롱폼 영상","쇼츠 제외하고 영상","쇼츠가 아닌 일반 영상","쇼츠는 빼고 영상","not shorts, long-form video","without Shorts, video"]:
   command="유튜브에서 아이언맨 "+wording+" 조회수 100만 이상 재생해줘"
   with self.subTest(wording=wording):
    self.assertEqual(youtube_content_kind(command),"video")
    self.assertTrue(youtube_video_request(command))
    self.assertEqual(youtube_format_requirement(command),{"required":"video","excluded":["shorts"]})
    self.assertEqual(view_requirement(command)["value"],1000000)
 def test_positive_shorts_and_reverse_exclusion_still_work(self):
  for wording in ["쇼츠 틀어줘","롱폼 말고 쇼츠 재생해줘","일반 영상 말고 shorts 재생"]:
   command="유튜브 "+wording
   with self.subTest(wording=wording):
    self.assertEqual(youtube_content_kind(command),"shorts")
    self.assertFalse(youtube_video_request(command))
 def test_playlist_and_live_exclusions_also_respect_negation(self):
  for wording in ["플레이리스트 말고 롱폼 영상","라이브 제외하고 일반 영상","not playlist, video","재생목록 말고 영상"]:
   command="유튜브에서 "+wording+" 재생"
   with self.subTest(wording=wording):self.assertEqual(youtube_content_kind(command),"video")
 def test_format_condition_is_visible_in_context_and_not_invented_for_unspecified_video(self):
  context=youtube_context("유튜브 아이언맨 쇼츠 말고 영상 조회수 100만 이상 재생",None)
  self.assertEqual(context["format_requirement"]["excluded"],["shorts"])
  self.assertIsNone(youtube_format_requirement("유튜브에서 아이언맨 영상 아무거나 재생"))
