"""
AG-UI Protocol Demo Server (Zero Extra Dependencies)
=====================================================
Manually implements the AG-UI SSE event protocol using only FastAPI + uvicorn.
Calls Bedrock Claude via boto3 (streaming) to demonstrate real AG-UI events.

No ag-ui-strands or strands-agents needed -- pure protocol implementation.

AG-UI Event Types implemented:
  RUN_STARTED, RUN_FINISHED
  TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END
  TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END
  STATE_SNAPSHOT

Run:
  python3 agui_server.py

Open:
  http://localhost:8080
"""

import ast
import json
import operator
import uuid
import asyncio
from typing import AsyncGenerator

import boto3
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# AG-UI Event helpers
# ---------------------------------------------------------------------------
def sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def agui_run_started(run_id: str) -> str:
    return sse_event({"type": "RUN_STARTED", "runId": run_id})


def agui_run_finished(run_id: str) -> str:
    return sse_event({"type": "RUN_FINISHED", "runId": run_id})


def agui_state_snapshot(state: dict) -> str:
    return sse_event({"type": "STATE_SNAPSHOT", "snapshot": state})


def agui_text_message_start(message_id: str, role: str = "assistant") -> str:
    return sse_event({"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": role})


def agui_text_message_content(message_id: str, delta: str) -> str:
    return sse_event({"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": delta})


def agui_text_message_end(message_id: str) -> str:
    return sse_event({"type": "TEXT_MESSAGE_END", "messageId": message_id})


def agui_tool_call_start(tool_call_id: str, tool_name: str) -> str:
    return sse_event({"type": "TOOL_CALL_START", "toolCallId": tool_call_id, "toolCallName": tool_name})


def agui_tool_call_args(tool_call_id: str, delta: str) -> str:
    return sse_event({"type": "TOOL_CALL_ARGS", "toolCallId": tool_call_id, "delta": delta})


def agui_tool_call_end(tool_call_id: str) -> str:
    return sse_event({"type": "TOOL_CALL_END", "toolCallId": tool_call_id})


# ---------------------------------------------------------------------------
# Safe math evaluator (no eval!)
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_calc(expr: str) -> str:
    """Evaluate arithmetic expressions without eval(). Supports +, -, *, /, parentheses."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result):
            return f"{expr} = {int(result)}"
        return f"{expr} = {result}"
    except (ValueError, TypeError, ZeroDivisionError, SyntaxError) as e:
        return f"Error evaluating '{expr}': {e}"


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe arithmetic operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Mock tools (demonstrate tool call visualization in AG-UI)
# ---------------------------------------------------------------------------
TOOLS = {
    "get_weather": {
        "description": "Get current weather for a city",
        "handler": lambda city: {
            "beijing": "Sunny, 28C, humidity 45%",
            "shanghai": "Cloudy, 25C, humidity 70%",
            "tokyo": "Rainy, 22C, humidity 85%",
            "new york": "Clear, 30C, humidity 50%",
            "london": "Overcast, 18C, humidity 75%",
            "seattle": "Drizzle, 15C, humidity 80%",
        }.get(city.lower(), f"No data for {city}"),
    },
    "calculate": {
        "description": "Evaluate a math expression safely",
        "handler": lambda expr: _safe_calc(expr),
    },
}

BEDROCK_TOOL_CONFIG = {
    "tools": [
        {"toolSpec": {"name": "get_weather", "description": "Get current weather for a city. Input: city name string.",
            "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}},
        {"toolSpec": {"name": "calculate", "description": "Evaluate a math expression. Input: expression string like '2+3*4'.",
            "inputSchema": {"json": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression"}}, "required": ["expression"]}}}},
    ]
}

# ---------------------------------------------------------------------------
# Bedrock Claude streaming with tool use
# ---------------------------------------------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

SYSTEM_PROMPT = (
    "You are a helpful assistant. You can check weather and do math calculations. "
    "When asked about weather, use the get_weather tool. "
    "When asked to calculate, use the calculate tool. "
    "Keep responses concise and friendly. Reply in the same language as the user."
)


async def agui_stream(input_data: dict) -> AsyncGenerator[str, None]:
    """
    Core AG-UI streaming logic:
    1. Send RUN_STARTED
    2. Call Bedrock Claude (streaming)
    3. Emit TEXT_MESSAGE_CONTENT / TOOL_CALL events as they arrive
    4. If tool_use, execute tool, feed result back, continue
    5. Send RUN_FINISHED
    """
    run_id = input_data.get("runId", str(uuid.uuid4()))
    messages_in = input_data.get("messages", [])

    bedrock_messages = []
    for m in messages_in:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant"):
            bedrock_messages.append({"role": role, "content": [{"text": content}]})
    if not bedrock_messages:
        bedrock_messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    yield agui_run_started(run_id)
    yield agui_state_snapshot({"status": "running"})

    for _round in range(5):
        msg_id = f"msg-{uuid.uuid4().hex[:8]}"
        try:
            response = bedrock.converse_stream(
                modelId=MODEL_ID, messages=bedrock_messages,
                system=[{"text": SYSTEM_PROMPT}], toolConfig=BEDROCK_TOOL_CONFIG,
            )
        except Exception:
            yield agui_text_message_start(msg_id)
            yield agui_text_message_content(msg_id, "Sorry, an error occurred while processing your request.")
            yield agui_text_message_end(msg_id)
            break

        text_started = False
        full_text = ""
        tool_use_blocks = []
        current_tool_id = None
        current_tool_name = None
        current_tool_args_json = ""

        for event in response.get("stream", []):
            if "contentBlockStart" in event:
                block = event["contentBlockStart"].get("start", {})
                if "toolUse" in block:
                    tu = block["toolUse"]
                    current_tool_id = tu.get("toolUseId", f"tool-{uuid.uuid4().hex[:8]}")
                    current_tool_name = tu.get("name", "unknown")
                    current_tool_args_json = ""
                    yield agui_tool_call_start(current_tool_id, current_tool_name)
                elif not text_started:
                    yield agui_text_message_start(msg_id)
                    text_started = True

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    chunk = delta["text"]
                    full_text += chunk
                    if not text_started:
                        yield agui_text_message_start(msg_id)
                        text_started = True
                    yield agui_text_message_content(msg_id, chunk)
                    await asyncio.sleep(0)
                elif "toolUse" in delta:
                    args_chunk = delta["toolUse"].get("input", "")
                    current_tool_args_json += args_chunk
                    yield agui_tool_call_args(current_tool_id, args_chunk)

            elif "contentBlockStop" in event:
                if current_tool_id:
                    try:
                        tool_args = json.loads(current_tool_args_json) if current_tool_args_json else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                    yield agui_tool_call_end(current_tool_id)
                    tool_use_blocks.append({"toolUseId": current_tool_id, "name": current_tool_name, "args": tool_args})
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_args_json = ""

        if text_started:
            yield agui_text_message_end(msg_id)
        if not tool_use_blocks:
            break

        assistant_content = []
        if full_text:
            assistant_content.append({"text": full_text})
        for tb in tool_use_blocks:
            assistant_content.append({"toolUse": {"toolUseId": tb["toolUseId"], "name": tb["name"], "input": tb["args"]}})
        bedrock_messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for tb in tool_use_blocks:
            handler = TOOLS.get(tb["name"], {}).get("handler")
            if handler:
                arg_val = list(tb["args"].values())[0] if tb["args"] else ""
                result_text = handler(arg_val)
            else:
                result_text = f"Unknown tool: {tb['name']}"
            tool_results.append({"toolResult": {"toolUseId": tb["toolUseId"], "content": [{"text": result_text}]}})
        bedrock_messages.append({"role": "user", "content": tool_results})

    yield agui_state_snapshot({"status": "completed"})
    yield agui_run_finished(run_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AG-UI Demo Server")

# NOTE: Open CORS for local demo only. Restrict origins in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/invocations")
async def invocations(input_data: dict, request: Request):
    """AG-UI SSE endpoint."""
    return StreamingResponse(agui_stream(input_data), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.get("/ping")
async def ping():
    return JSONResponse({"status": "Healthy"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND_HTML


FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AG-UI Protocol Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f1117;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
.header{background:#1a1b26;border-bottom:1px solid #2a2b3d;padding:16px 24px;display:flex;align-items:center;gap:12px}
.header h1{font-size:18px;font-weight:600;color:#7aa2f7}
.header .badge{background:#1a3a2a;color:#9ece6a;font-size:11px;padding:3px 8px;border-radius:10px;font-weight:500}
.header .info{margin-left:auto;font-size:12px;color:#565f89}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.6;font-size:14px;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#283457;color:#c0caf5;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#1a1b26;color:#c0caf5;border:1px solid #2a2b3d;border-bottom-left-radius:4px}
.msg.assistant.streaming{border-color:#7aa2f7;box-shadow:0 0 8px rgba(122,162,247,0.15)}
.evt{align-self:center;font-size:11px;padding:3px 12px;border-radius:8px;font-family:'SF Mono','Fira Code',monospace}
.evt.run{background:#1a2a1a;color:#9ece6a}.evt.tool{background:#2a2a1a;color:#e0af68}
.evt.state{background:#1a1a2a;color:#7aa2f7}.evt.err{background:#2a1a1a;color:#f7768e}
.input-area{background:#1a1b26;border-top:1px solid #2a2b3d;padding:16px 24px;display:flex;gap:12px;align-items:center}
.input-area input{flex:1;background:#24283b;border:1px solid #2a2b3d;color:#c0caf5;padding:12px 16px;border-radius:8px;font-size:14px;outline:none}
.input-area input:focus{border-color:#7aa2f7}
.input-area input::placeholder{color:#565f89}
.input-area button{background:#7aa2f7;color:#1a1b26;border:none;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.input-area button:hover{opacity:0.85}.input-area button:disabled{opacity:0.4;cursor:not-allowed}
.quick{display:flex;gap:8px;padding:8px 24px;flex-wrap:wrap}
.qbtn{background:#24283b;border:1px solid #2a2b3d;color:#7aa2f7;padding:6px 14px;border-radius:16px;font-size:12px;cursor:pointer}
.qbtn:hover{background:#283457;border-color:#7aa2f7}
.chat::-webkit-scrollbar{width:6px}.chat::-webkit-scrollbar-track{background:transparent}
.chat::-webkit-scrollbar-thumb{background:#2a2b3d;border-radius:3px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.cursor::after{content:'|';animation:blink 0.8s infinite;color:#7aa2f7;margin-left:1px}
</style>
</head>
<body>
<div class="header"><h1>AG-UI Protocol Demo</h1><span class="badge">SSE</span><span class="info">Bedrock Claude | FastAPI | Port 8080</span></div>
<div class="chat" id="chat">
  <div class="evt state">-- Welcome! This demo implements the AG-UI event protocol with Bedrock Claude --</div>
  <div class="evt state">-- Events: RUN_STARTED / TEXT_MESSAGE / TOOL_CALL / RUN_FINISHED --</div>
</div>
<div class="quick">
  <button class="qbtn" onclick="q('What is the AG-UI protocol? Explain briefly.')">What is AG-UI?</button>
  <button class="qbtn" onclick="q('Check weather in Beijing and Tokyo')">Weather: Beijing + Tokyo</button>
  <button class="qbtn" onclick="q('Calculate (3.14 * 100) + (2.718 * 50)')">Math calculation</button>
  <button class="qbtn" onclick="q('What is 42 * 17? Also check weather in Seattle.')">Tools combo</button>
</div>
<div class="input-area">
  <input id="inp" placeholder="Ask anything... try weather or math for tool demos" onkeydown="if(event.key==='Enter')send()" />
  <button id="btn" onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),btn=document.getElementById('btn');
let n=0;function uid(){return 'id-'+(++n)+'-'+Math.random().toString(36).slice(2,7)}
function addEvt(t,c){const e=document.createElement('div');e.className='evt '+c;e.textContent=t;chat.appendChild(e);chat.scrollTop=chat.scrollHeight}
function addMsg(r,t){const e=document.createElement('div');e.className='msg '+r;e.textContent=t||'';chat.appendChild(e);chat.scrollTop=chat.scrollHeight;return e}
function q(t){inp.value=t;send()}
async function send(){const t=inp.value.trim();if(!t)return;inp.value='';btn.disabled=true;addMsg('user',t);
const body={threadId:'t-'+Date.now(),runId:uid(),state:{},messages:[{role:'user',content:t,id:uid()}],tools:[],context:[],forwardedProps:{}};
let el=null,txt='';try{const r=await fetch('/invocations',{method:'POST',headers:{'Content-Type':'application/json','Accept':'text/event-stream'},body:JSON.stringify(body)});
const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
while(true){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const lines=buf.split('\n');buf=lines.pop();
for(const ln of lines){if(!ln.startsWith('data: '))continue;try{handle(JSON.parse(ln.slice(6)))}catch(e){}}}
if(buf.startsWith('data: ')){try{handle(JSON.parse(buf.slice(6)))}catch(e){}}}catch(e){addEvt('ERROR: '+e.message,'err')}
btn.disabled=false;inp.focus();
function handle(ev){switch(ev.type){
case 'RUN_STARTED':addEvt('>> RUN_STARTED','run');break;
case 'RUN_FINISHED':addEvt('<< RUN_FINISHED','run');if(el)el.classList.remove('streaming','cursor');break;
case 'STATE_SNAPSHOT':addEvt('-- STATE: '+JSON.stringify(ev.snapshot||{}),'state');break;
case 'TEXT_MESSAGE_START':txt='';el=addMsg('assistant','');el.classList.add('streaming','cursor');break;
case 'TEXT_MESSAGE_CONTENT':if(ev.delta&&el){txt+=ev.delta;el.textContent=txt;chat.scrollTop=chat.scrollHeight}break;
case 'TEXT_MESSAGE_END':if(el)el.classList.remove('streaming','cursor');break;
case 'TOOL_CALL_START':addEvt('>> TOOL: '+ev.toolCallName+'()','tool');break;
case 'TOOL_CALL_ARGS':if(ev.delta)addEvt('   args: '+ev.delta,'tool');break;
case 'TOOL_CALL_END':addEvt('<< TOOL done','tool');break;
default:if(ev.type)addEvt('-- '+ev.type,'state')}}}
inp.focus();
</script>
</body></html>
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  AG-UI Protocol Demo Server")
    print("  Web UI:  http://localhost:8080")
    print("  SSE:     POST http://localhost:8080/invocations")
    print("  Health:  GET  http://localhost:8080/ping")
    print("=" * 60)
    # NOTE: Binds to all interfaces for local demo. Use 127.0.0.1 in production.
    uvicorn.run(app, host="0.0.0.0", port=8080)
