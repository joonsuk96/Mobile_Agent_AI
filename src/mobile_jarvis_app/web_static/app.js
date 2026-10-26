"use strict";
const $ = id => document.getElementById(id);
let current = null, cursor = 0, busy = false, connected = false, started = null, count = 0, hiddenTask = null, epoch = 0;
const restoredTaskIds = new Set();
const terminal = new Set(["finished", "cancelled", "error"]);
const names = {running:["명령 수행 중", "EXECUTING"], waiting:["답변을 기다리고 있어요", "AWAITING INPUT"], cancelling:["실행 중단 중", "STOPPING"], finished:["작업 종료", "FINISHED"], cancelled:["작업 중단됨", "CANCELLED"], error:["실행 확인 필요", "CHECK REQUIRED"]};
function toast(text) { $("toast").textContent = text; $("toast").hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => $("toast").hidden = true, 6000); }
async function api(path, body) {
  const options = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-Jarvis-Client":"web"}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(typeof data.detail === "string" ? data.detail : "요청을 확인해 주세요. 입력값 또는 서버 상태를 확인해 주세요."); }
  return response.json();
}
function message(text, type="assistant") {
  const area = $("chat-area"), row = document.createElement("div"); row.className = "message " + type;
  if (type === "user") { const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text; row.append(bubble); }
  else { const avatar = document.createElement("span"); avatar.className = "assistant-avatar"; avatar.textContent = "✧"; const body = document.createElement("div"); const name = document.createElement("span"); name.className = "message-name"; name.textContent = "JARVIS"; const p = document.createElement("p"); p.textContent = text.replace(/^자비스\s*:\s*/, ""); body.append(name,p); row.append(avatar,body); }
  area.append(row); area.scrollTop = area.scrollHeight;
}
function log(event) {
  const area = $("activity-log"); const atBottom = area.scrollHeight - area.scrollTop - area.clientHeight < 60;
  area.querySelector(".empty-log")?.remove();
  const item = document.createElement("div"); item.className = "log-item" + (event.text.includes("[행동") ? " action" : "") + (event.type === "error" ? " error" : "");
  const time = document.createElement("small"); time.textContent = new Date().toLocaleTimeString("ko-KR", {hour12:false}); const text = document.createElement("p"); text.textContent = event.text; item.append(time,text); area.append(item);
  if (area.childElementCount > 300) area.firstElementChild.remove();
  $("event-count").textContent = String(++count).padStart(2,"0"); if (atBottom) area.scrollTop = area.scrollHeight;
}
function render() {
  const active = current && !terminal.has(current.status), waiting = current?.status === "waiting";
  const [label, code] = current ? (names[current.status] || names.error) : ["명령 대기 중", "STANDBY"];
  $("task-status").textContent = label; $("core-status").textContent = code; $("live-tag").textContent = active ? (waiting ? "WAIT" : "LIVE") : "IDLE";
  $("stage-message").textContent = current ? (waiting ? "다음 단계를 위해 답변이 필요해요." : active ? "목표를 이해하고, 다음 행동을 실행하고 있어요." : "작업이 종료됐어요. 다음 명령을 기다릴게요.") : "준비됐어요. 무엇을 도와드릴까요?";
  $("stage").classList.toggle("is-running", !!active && !waiting); $("status-orb").classList.toggle("active", !!active);
  $("command").disabled = busy || (!!active && !waiting); $("send").disabled = busy || !connected || (!!active && !waiting); $("cancel").disabled = busy || !active || current.status === "cancelling";
  $("new-chat").disabled = !!active; $("input-mode").textContent = waiting ? "YOUR ANSWER" : "COMMAND INPUT";
  $("command").placeholder = waiting ? "자비스의 질문에 답해 주세요…" : active ? "자비스가 실행 중입니다. 진행 상황을 확인해 주세요…" : "자비스, 유튜브에서 좋아하는 음악 틀어줘…";
  document.querySelectorAll("[data-prompt]").forEach(button => button.disabled = !!active || busy);
}
function consume(data) {
  const task = data.task;
  if (!task || task.id === hiddenTask) { render(); return; }
  if (current?.id === task.id && task.cursor < cursor) return;
  if (!current || current.id !== task.id) {
    current = task; cursor = 0; started = Date.now();
    if (!restoredTaskIds.has(task.id)) message(task.command,"user"); $("suggestions").hidden = true;
  }
  const wasWaiting = current.status === "waiting";
  current = task;
  for (const event of task.events || []) {
    if (event.id <= cursor) continue; cursor = event.id;
    if (event.type === "answer" && !restoredTaskIds.has(task.id)) message(event.text);
    else if (event.type === "question" && !restoredTaskIds.has(task.id)) message(event.text,"question");
    else if (event.type === "reply" && !restoredTaskIds.has(task.id)) message(event.text,"user");
    else if (event.type === "error") { log(event); message(event.text,"error"); }
    else if (event.type === "log") log(event);
    else if (event.type === "status" && event.text === "cancelled") message("작업을 중단했어요. 이미 폰에 전달된 동작은 완료될 수 있어요.");
  }
  render(); if (task.status === "waiting" && !wasWaiting) $("command").focus();
}
async function poll() {
  const requestEpoch = epoch;
  try { const data = await api(`/api/state?after=${cursor}&task_id=${encodeURIComponent(current?.id || "")}`); connected = true; $("server-badge").classList.remove("offline"); $("server-badge").querySelector("span").textContent = "서버 연결됨"; if (requestEpoch === epoch) consume(data); }
  catch { connected = false; $("server-badge").classList.add("offline"); $("server-badge").querySelector("span").textContent = "서버 재연결 중"; render(); }
  finally { setTimeout(poll, 800); }
}
$("command-form").addEventListener("submit", async event => {
  event.preventDefault(); const text = $("command").value.trim(); if (!text || busy) return;
  const reply = current?.status === "waiting";
  if (current && !terminal.has(current.status) && !reply) return;
  busy = true; epoch++; render();
  try { hiddenTask = null; consume(await api(reply ? `/api/tasks/${current.id}/reply` : "/api/tasks", {text})); $("command").value = ""; }
  catch (error) { toast(error.message); }
  finally { busy = false; render(); }
});
$("command").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("command-form").requestSubmit(); } });
$("cancel").addEventListener("click", async () => { if (!current || busy) return; busy = true; epoch++; render(); try { consume(await api(`/api/tasks/${current.id}/cancel`, {})); } catch (error) { toast(error.message); } finally { busy = false; render(); } });
$("new-chat").addEventListener("click", async () => {
  if (busy || current && !terminal.has(current.status)) return;
  busy = true; epoch++; render();
  try {
    await api("/api/conversation/reset", {});
    hiddenTask = null; current = null; cursor = 0; count = 0; started = null;
    restoredTaskIds.clear(); $("chat-area").replaceChildren();
    message("새 대화를 시작했어요. 이전 대화 기억을 비웠어요. 무엇을 해드릴까요?");
    $("activity-log").replaceChildren(); $("event-count").textContent = "00"; $("elapsed").textContent = "00:00";
    $("suggestions").hidden = false; $("command").value = "";
  } catch (error) { toast(error.message); }
  finally { busy = false; render(); $("command").focus(); }
});

document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => { $("command").value = button.dataset.prompt; $("command").focus(); }));
setInterval(() => { if (!started || !current || terminal.has(current.status)) return; const seconds = Math.floor((Date.now()-started)/1000); $("elapsed").textContent = `${String(Math.floor(seconds/60)).padStart(2,"0")}:${String(seconds%60).padStart(2,"0")}`; },1000);
async function boot() {
  try {
    const data = await api("/api/conversation");
    if (data.turns.length) {
      $("chat-area").replaceChildren();
      for (const turn of data.turns) {
        restoredTaskIds.add(turn.id); message(turn.command, "user");
        message(turn.answer || (turn.status === "cancelled" ? "이 작업은 중단됐어요." : "이 작업의 최종 답변은 기록되지 않았어요."));
      }
    }
  } catch { /* Polling below exposes the connection state. */ }
  poll();
}
render(); boot();
