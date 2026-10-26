"""Web transport tests use a fake child process, never the phone or paid API."""
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from fastapi.testclient import TestClient
from mobile_jarvis_app.web_server import TaskManager, create_app

FAKE = r"""
import json,sys,time
command=json.loads(sys.stdin.readline())["command"]
def emit(kind,text):
 print(json.dumps({"type":kind,"text":text},ensure_ascii=False),flush=True)
emit("log","[행동 1/100] 테스트 전용 실행")
if command=="wait":
 time.sleep(30)
else:
 emit("question","인원은 몇 명인가요?")
 answer=json.loads(sys.stdin.readline())["answer"]
 emit("answer","자비스 : 테스트 답변 "+answer)
"""
HEADERS={"X-Jarvis-Client":"web"}

class WebConsoleTests(unittest.TestCase):
 def setUp(self):
  self.manager=TaskManager([sys.executable,"-u","-c",FAKE])
  self.client=TestClient(create_app(self.manager))
  self.client.__enter__()
  self.addCleanup(self.client.__exit__,None,None,None)
 def post(self,path,**data):
  return self.client.post(path,json=data,headers=HEADERS)
 def wait_status(self,status):
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   task=self.manager.snapshot()["task"]
   if task and task["status"]==status:return task
   time.sleep(.02)
  self.fail(f"Expected {status}: {self.manager.snapshot()}")
 def test_home_and_assets_no_agent_execution(self):
  response=self.client.get("/")
  self.assertEqual(response.status_code,200)
  self.assertIn("JARVIS CORE",response.text)
  for name in ("app.js","style.css"):
   self.assertEqual(self.client.get("/assets/"+name).status_code,200)
  self.assertIsNone(self.manager.task)
 def test_command_question_answer_and_cursor(self):
  response=self.post("/api/tasks",text="테스트 명령")
  self.assertEqual(response.status_code,202)
  task=self.wait_status("waiting")
  self.assertEqual(task["question"],"인원은 몇 명인가요?")
  cursor=task["cursor"]
  self.assertEqual(self.post("/api/tasks",text="겹치는 작업").status_code,409)
  response=self.post(f"/api/tasks/{task['id']}/reply",text="2명")
  self.assertEqual(response.status_code,200)
  task=self.wait_status("finished")
  self.assertEqual(task["answer"],"자비스 : 테스트 답변 2명")
  events=self.client.get(f"/api/state?task_id={task['id']}&after={cursor}").json()["task"]["events"]
  self.assertTrue(all(event["id"]>cursor for event in events))
  self.assertEqual(self.post(f"/api/tasks/{task['id']}/reply",text="늦은 답변").status_code,409)
 def test_cancel_terminates_worker_and_allows_new_task(self):
  task=self.post("/api/tasks",text="wait").json()["task"]
  self.assertEqual(self.post(f"/api/tasks/{task['id']}/cancel").json()["task"]["status"],"cancelled")
  self.assertIsNotNone(self.manager.process.poll())
  self.assertEqual(self.post("/api/tasks",text="다음 작업").status_code,202)
  self.wait_status("waiting")
 def test_rejects_blank_foreign_origin_and_unknown_task(self):
  self.assertEqual(self.post("/api/tasks",text="   ").status_code,422)
  self.assertEqual(self.client.post("/api/tasks",json={"text":"안됨"}).status_code,403)
  self.assertEqual(self.client.post("/api/tasks",json={"text":"안됨"},headers={**HEADERS,"Origin":"https://example.com"}).status_code,403)
  self.assertEqual(self.post("/api/tasks/missing/cancel").status_code,404)
 def test_refresh_receives_retained_events(self):
  self.post("/api/tasks",text="테스트")
  task=self.wait_status("waiting")
  restored=self.client.get("/api/state?task_id=unknown&after=100").json()["task"]
  self.assertEqual(restored["id"],task["id"])
  self.assertTrue(restored["events"])
 def test_worker_without_final_answer_is_not_reported_as_success(self):
  self.manager.worker_command=[sys.executable,"-c", "print('not a protocol message')"]
  self.post("/api/tasks",text="테스트")
  task=self.wait_status("error")
  self.assertIsNone(task["answer"])
  self.assertTrue(any(event["type"]=="error" for event in task["events"]))
 def test_same_origin_header_request_is_accepted(self):
  response=self.client.post("/api/tasks",json={"text":"테스트"},headers={**HEADERS,"Origin":"http://testserver"})
  self.assertEqual(response.status_code,202)
  self.wait_status("waiting")

class WorkerAdapterTests(unittest.TestCase):
 def test_input_and_output_bridge_keeps_existing_terminal_behavior(self):
  script="""
from mobile_jarvis_app import agent
from mobile_jarvis_app.web_worker import run
def fake():
 print('AI: 몇 명인가요?')
 result=input('답변: ')
 print('자비스 : '+result)
agent.main=fake
run()
"""
  result=subprocess.run([sys.executable,"-u","-c",script],input=json.dumps({"command":"테스트"})+"\n"+json.dumps({"answer":"2명"})+"\n",capture_output=True,text=True,encoding="utf-8",timeout=20)
  self.assertEqual(result.returncode,0,result.stderr)
  events=[json.loads(line) for line in result.stdout.splitlines()]
  self.assertIn({"type":"question","text":"몇 명인가요?","prompt":"답변: "},events)
  self.assertIn({"type":"answer","text":"자비스 : 2명"},events)
