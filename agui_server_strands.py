"""
AG-UI Demo Server -- Strands Agent Version
===========================================
Uses ag-ui-strands to automatically convert Strands Agent events
into AG-UI protocol events. Minimal code, zero manual SSE formatting.

Compare with agui_server.py (manual implementation) to see the difference.

Run:
  source .venv/bin/activate
  python agui_server_strands.py

Open:
  http://localhost:8090
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from ag_ui_strands import StrandsAgent
from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from strands import Agent, tool
from strands.models.bedrock import BedrockModel


# ---------------------------------------------------------------------------
# 1. Define tools (same as manual version, for comparison)
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "beijing": "Sunny, 28C, humidity 45%",
        "shanghai": "Cloudy, 25C, humidity 70%",
        "tokyo": "Rainy, 22C, humidity 85%",
        "new york": "Clear, 30C, humidity 50%",
        "london": "Overcast, 18C, humidity 75%",
        "seattle": "Drizzle, 15C, humidity 80%",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely. Supports +, -, *, /, parentheses."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        if isinstance(result, float) and result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {result}"
    except (ValueError, TypeError, ZeroDivisionError, SyntaxError) as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# 2. Create Strands Agent (standard, nothing AG-UI specific)
# ---------------------------------------------------------------------------
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-west-2",
)

strands_agent = Agent(
    model=model,
    system_prompt=(
        "You are a helpful assistant. You can check weather and do calculations. "
        "When asked about weather, use the get_weather tool. "
        "When asked to calculate, use the calculate tool. "
        "Keep responses concise and friendly. Reply in the same language as the user."
    ),
    tools=[get_weather, calculate],
)

# ---------------------------------------------------------------------------
# 3. ONE LINE: wrap as AG-UI agent (this is the entire protocol adaptation)
# ---------------------------------------------------------------------------
agui_agent = StrandsAgent(
    agent=strands_agent,
    name="strands_demo_agent",
    description="A demo agent with weather and calculator tools, powered by Strands + Bedrock Claude",
)

# ---------------------------------------------------------------------------
# 4. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AG-UI Demo (Strands)")

# NOTE: Open CORS for local demo only. Restrict origins in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/invocations")
async def invocations(input_data: dict, request: Request):
    """AG-UI SSE endpoint -- StrandsAgent handles all event formatting."""
    accept_header = request.headers.get("accept", "text/event-stream")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        run_input = RunAgentInput(**input_data)
        async for event in agui_agent.run(run_input):
            yield encoder.encode(event)

    return StreamingResponse(event_generator(), media_type=encoder.get_content_type(),
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
<title>AG-UI Demo (Strands)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f1117;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
.header{background:#1a1b26;border-bottom:1px solid #2a2b3d;padding:16px 24px;display:flex;align-items:center;gap:12px}
.header h1{font-size:18px;font-weight:600;color:#bb9af7}
.header .badge{background:#2a1a3a;color:#bb9af7;font-size:11px;padding:3px 8px;border-radius:10px;font-weight:500}
.header .info{margin-left:auto;font-size:12px;color:#565f89}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.6;font-size:14px;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#34284a;color:#c0caf5;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#1a1b26;color:#c0caf5;border:1px solid #2a2b3d;border-bottom-left-radius:4px}
.msg.assistant.streaming{border-color:#bb9af7;box-shadow:0 0 8px rgba(187,154,247,0.15)}
.evt{align-self:center;font-size:11px;padding:3px 12px;border-radius:8px;font-family:'SF Mono','Fira Code',monospace}
.evt.run{background:#1a2a1a;color:#9ece6a}.evt.tool{background:#2a2a1a;color:#e0af68}
.evt.state{background:#1a1a2a;color:#7aa2f7}.evt.err{background:#2a1a1a;color:#f7768e}
.input-area{background:#1a1b26;border-top:1px solid #2a2b3d;padding:16px 24px;display:flex;gap:12px;align-items:center}
.input-area input{flex:1;background:#24283b;border:1px solid #2a2b3d;color:#c0caf5;padding:12px 16px;border-radius:8px;font-size:14px;outline:none}
.input-area input:focus{border-color:#bb9af7}
.input-area input::placeholder{color:#565f89}
.input-area button{background:#bb9af7;color:#1a1b26;border:none;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.input-area button:hover{opacity:0.85}.input-area button:disabled{opacity:0.4;cursor:not-allowed}
.quick{display:flex;gap:8px;padding:8px 24px;flex-wrap:wrap}
.qbtn{background:#24283b;border:1px solid #2a2b3d;color:#bb9af7;padding:6px 14px;border-radius:16px;font-size:12px;cursor:pointer}
.qbtn:hover{background:#34284a;border-color:#bb9af7}
.chat::-webkit-scrollbar{width:6px}.chat::-webkit-scrollbar-track{background:transparent}
.chat::-webkit-scrollbar-thumb{background:#2a2b3d;border-radius:3px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.cursor::after{content:'|';animation:blink 0.8s infinite;color:#bb9af7;margin-left:1px}
</style>
</head>
<body>
<div class="header"><h1>AG-UI Demo (Strands Agent)</h1><span class="badge">ag-ui-strands</span><span class="info">Strands + Bedrock Claude | Port 8090</span></div>
<div class="chat" id="chat">
  <div class="evt state">-- Strands version: ag-ui-strands auto-converts Agent events to AG-UI protocol --</div>
  <div class="evt state">-- Compare with manual version on port 8080 --</div>
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
case 'STATE_SNAPSHOT':addEvt('-- STATE_SNAPSHOT','state');break;
case 'STATE_DELTA':addEvt('-- STATE_DELTA: '+JSON.stringify(ev.delta||''),'state');break;
case 'TEXT_MESSAGE_START':txt='';el=addMsg('assistant','');el.classList.add('streaming','cursor');break;
case 'TEXT_MESSAGE_CONTENT':if(ev.delta&&el){txt+=ev.delta;el.textContent=txt;chat.scrollTop=chat.scrollHeight}break;
case 'TEXT_MESSAGE_END':if(el)el.classList.remove('streaming','cursor');break;
case 'TOOL_CALL_START':addEvt('>> TOOL: '+(ev.toolCallName||ev.name||'?')+'()','tool');break;
case 'TOOL_CALL_ARGS':if(ev.delta)addEvt('   args: '+ev.delta,'tool');break;
case 'TOOL_CALL_END':addEvt('<< TOOL done','tool');break;
default:if(ev.type)addEvt('-- '+ev.type,'state')}}}
inp.focus();
</script>
</body></html>
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  AG-UI Demo Server (Strands Version)")
    print("  Web UI:  http://localhost:8090")
    print("  SSE:     POST http://localhost:8090/invocations")
    print("  Health:  GET  http://localhost:8090/ping")
    print("")
    print("  Compare with manual version: http://localhost:8080")
    print("=" * 60)
    # NOTE: Binds to all interfaces for local demo. Use 127.0.0.1 in production.
    uvicorn.run(app, host="0.0.0.0", port=8090)
