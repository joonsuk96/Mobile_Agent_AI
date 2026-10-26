"""Bounded web history, follow-up interpretation, and conversation-only replies."""
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from mobile_jarvis_app.conversation import resolve_turn
from mobile_jarvis_app.web_server import TaskManager, create_app

class ResolveTurnTests(unittest.TestCase):
 def test_history_is_sent_to_model_and_followup_becomes_independent_command(self):
  model=Mock();model.invoke.return_value=AIMessage(content=json.dumps({"mode":"action","command":"유튜브에서 아이언맨 조회수 200만 이상 영상 재생해줘"},ensure_ascii=False))
  history=[{"command":"유튜브에서 아이언맨 조회수 100만 이상 영상 재생","answer":"실행했어","status":"finished","verified_success":True}]
  result=resolve_turn(model,"이번에는 200만 이상으로",history)
  self.assertIn("200만",result.command)
  payload=json.loads(model.invoke.call_args.args[0][1].content)
  self.assertEqual(payload["previous_turns"],history)
  self.assertEqual(payload["current_message"],"이번에는 200만 이상으로")
 def test_plain_conversation_and_clarification_are_not_actions(self):
  for mode in ("answer","clarify"):
   model=Mock();model.invoke.return_value=AIMessage(content=json.dumps({"mode":mode,"message":"직전에는 아이언맨 영상을 요청했어."},ensure_ascii=False))
   self.assertEqual(resolve_turn(model,"아까 뭐라고 했지?",[]).mode,mode)
 def test_invalid_output_does_not_become_a_phone_command(self):
  model=Mock();model.invoke.return_value=AIMessage(content="잘 모르겠어요")
  with self.assertRaises(ValueError):resolve_turn(model,"그거 해줘",[])
 def test_only_recent_twelve_turns_are_forwarded(self):
  model=Mock();model.invoke.return_value=AIMessage(content='{"mode":"answer","message":"기록 확인"}')
  resolve_turn(model,"기록",[{"command":str(i)} for i in range(25)])
  self.assertEqual(len(json.loads(model.invoke.call_args.args[0][1].content)["previous_turns"]),12)

FAKE=r"""
import json,sys
initial=json.loads(sys.stdin.readline())
print(json.dumps({"type":"answer","text":"자비스 : 이전 기록 "+str(len(initial.get('history',[])))},ensure_ascii=False),flush=True)
print(json.dumps({"type":"outcome","verified_success":True}),flush=True)
"""
class ServerMemoryTests(unittest.TestCase):
 def setUp(self):
  self.manager=TaskManager([sys.executable,"-u","-c",FAKE])
  self.client=TestClient(create_app(self.manager));self.client.__enter__()
  self.addCleanup(self.client.__exit__,None,None,None)
 def post(self,path,**body):return self.client.post(path,json=body,headers={"X-Jarvis-Client":"web"})
 def wait(self):
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   task=self.manager.snapshot()["task"]
   if task["status"]=="finished":return task
   time.sleep(.02)
  self.fail("task did not finish")
 def test_second_worker_receives_first_request_and_answer(self):
  self.post("/api/tasks",text="아이언맨 검색");self.wait()
  self.post("/api/tasks",text="전에 뭐라고 했지?")
  task=self.wait()
  self.assertIn("이전 기록 1",task["answer"])
  history=self.client.get("/api/conversation").json()["turns"]
  self.assertEqual([turn["command"] for turn in history],["아이언맨 검색","전에 뭐라고 했지?"])
  self.assertTrue(history[0]["verified_success"])
 def test_new_conversation_resets_server_memory_and_latest_task(self):
  self.post("/api/tasks",text="이전 명령");self.wait()
  self.assertEqual(self.post("/api/conversation/reset").status_code,200)
  self.assertEqual(self.client.get("/api/conversation").json()["turns"],[])
  self.assertIsNone(self.manager.task)
  self.post("/api/tasks",text="새 명령")
  self.assertIn("이전 기록 0",self.wait()["answer"])
 def test_history_is_bounded(self):
  for i in range(30):
   self.manager.task={"id":str(i),"command":str(i),"status":"finished","events":[],"sequence":0}
   self.manager._archive()
  self.assertEqual(len(self.manager.conversation),20)
  self.assertEqual(self.manager.conversation[0]["command"],"10")

class ConversationWorkerTests(unittest.TestCase):
 def test_history_question_returns_answer_without_starting_phone_agent(self):
  script="""
from mobile_jarvis_app import agent
from langchain_core.messages import AIMessage
class Model:
 def invoke(self,messages):
  return AIMessage(content='{"mode":"answer","message":"직전에는 아이언맨 영상을 요청했어."}')
agent.build_model=lambda:Model()
def forbidden():raise RuntimeError('phone agent must not start')
agent.main=forbidden
from mobile_jarvis_app.web_worker import run
run()
"""
  result=subprocess.run([sys.executable,"-u","-c",script],input=json.dumps({"command":"아까 뭐라고 했지?","history":[{"command":"아이언맨"}],"conversation_enabled":True})+"\n",capture_output=True,text=True,encoding="utf-8",timeout=20)
  self.assertEqual(result.returncode,0,result.stderr)
  events=[json.loads(line) for line in result.stdout.splitlines()]
  self.assertTrue(any(event["type"]=="answer" and "아이언맨" in event["text"] for event in events))
  self.assertFalse(any(event["type"]=="error" for event in events))
