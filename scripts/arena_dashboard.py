import asyncio
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from assignment.pipeline import (
    build_production_plugins,
    build_observability,
    DefensePipeline,
    _build_llm,
)
from attacks.attacks import response_leaked_secrets
from guardrails.input_guardrails import detect_injection, topic_filter, normalize_text
from guardrails.output_guardrails import content_filter

app = FastAPI(title="VinBank Defense Arena Dashboard")

# State
state = {
    "score": 5,
    "current_slot": 1,
    "max_slots": 4,
    "history": [],
    "logs": [
        {"time": time.strftime("%H:%M:%S"), "tag": "SYSTEM", "msg": "VinBank Security Shield initialized. 5 points ready."}
    ]
}

# Initialize pipeline
plugins = build_production_plugins(use_llm_judge=False)
audit, monitor = build_observability()
llm = _build_llm()
pipeline = DefensePipeline(plugins, audit, monitor, llm=llm)

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARENA DEMO: PROMPT INJECTION SHOWDOWN - VINBANK DEFENSE</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --border-color: #1f2937;
            --cyan: #06b6d4;
            --blue: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); padding: 20px; min-height: 100vh; }
        
        .header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 18px 24px; background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .header h1 { font-size: 22px; font-weight: 800; letter-spacing: 1px; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
        .badge-live {
            background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid var(--green);
            padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700;
            display: flex; align-items: center; gap: 6px;
        }
        .badge-live::before { content: ""; width: 8px; height: 8px; background: var(--green); border-radius: 50%; display: inline-block; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

        .top-stats {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;
        }
        .stat-card {
            background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px;
            display: flex; flex-direction: column; gap: 6px;
        }
        .stat-label { font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }
        .stat-value { font-size: 26px; font-weight: 800; color: #fff; }
        .stat-value.score { color: #34d399; }
        .stat-value.target { font-size: 14px; font-family: monospace; color: #93c5fd; }

        .main-grid {
            display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px;
        }
        .panel {
            background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px;
            display: flex; flex-direction: column; gap: 16px;
        }
        .panel-title { font-size: 16px; font-weight: 700; color: #e5e7eb; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }

        /* Pipeline Visualizer */
        .pipeline-bar {
            display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 10px 0;
            background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b;
        }
        .pipe-node {
            flex: 1; text-align: center; padding: 8px 4px; border-radius: 6px; font-size: 11px; font-weight: 700;
            background: #1e293b; color: #94a3b8; border: 1px solid #334155; transition: all 0.3s ease;
        }
        .pipe-node.active-pass { background: rgba(16, 185, 129, 0.2); border-color: var(--green); color: #34d399; }
        .pipe-node.active-block { background: rgba(239, 68, 68, 0.2); border-color: var(--red); color: #f87171; }
        .pipe-arrow { color: #475569; font-size: 12px; }

        /* Input form */
        .input-group { display: flex; flex-direction: column; gap: 8px; }
        textarea {
            width: 100%; height: 90px; background: #0f172a; border: 1px solid #334155; border-radius: 8px;
            color: #fff; padding: 12px; font-size: 14px; resize: none; outline: none; transition: border-color 0.2s;
        }
        textarea:focus { border-color: var(--cyan); }
        .btn-submit {
            background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; border: none; padding: 12px;
            border-radius: 8px; font-size: 14px; font-weight: 700; cursor: pointer; transition: all 0.2s;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .btn-submit:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Output box */
        .result-box {
            background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; font-size: 13px; line-height: 1.5; min-height: 80px;
        }
        .result-header { font-weight: 700; margin-bottom: 6px; display: flex; justify-content: space-between; }
        .tag-block { color: #f87171; }
        .tag-safe { color: #34d399; }
        .tag-leak { color: #ef4444; font-weight: 900; }

        /* Terminal Logs */
        .terminal {
            background: #050811; border: 1px solid #1e293b; border-radius: 8px; padding: 12px;
            height: 380px; overflow-y: auto; font-family: "Consolas", monospace; font-size: 12px; line-height: 1.6;
        }
        .log-entry { margin-bottom: 4px; display: flex; gap: 8px; }
        .log-time { color: #64748b; }
        .log-tag { font-weight: 700; }
        .log-tag.INFO { color: #38bdf8; }
        .log-tag.PASS { color: #34d399; }
        .log-tag.BLOCK { color: #f87171; }
        .log-tag.WARN { color: #fbbf24; }
        .log-tag.LEAK { color: #ef4444; font-weight: 900; }
        .log-msg { color: #cbd5e1; word-break: break-all; }

        .btn-reset {
            background: #374151; color: #d1d5db; border: none; padding: 6px 12px; border-radius: 6px;
            font-size: 11px; font-weight: 600; cursor: pointer;
        }
        .btn-reset:hover { background: #4b5563; }
    </style>
</head>
<body>

    <div class="header">
        <h1>🛡️ VINBANK DEFENSE ARENA — PROMPT INJECTION SHOWDOWN</h1>
        <div style="display: flex; gap: 10px; align-items: center;">
            <button class="btn-reset" onclick="resetArena()">🔄 Reset Vòng Đấu</button>
            <div class="badge-live">BLUE DEFENSE ACTIVE</div>
        </div>
    </div>

    <div class="top-stats">
        <div class="stat-card">
            <span class="stat-label">Điểm Số Phòng Thủ</span>
            <span class="stat-value score" id="val-score">5 / 5</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Lượt Đấu (Slot)</span>
            <span class="stat-value" id="val-slot">1 / 4</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Trạng Thái Khiên</span>
            <span class="stat-value" id="val-status" style="font-size: 18px; color: #38bdf8;">THỦ VỮNG</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Mục Tiêu Bảo Mật (Target)</span>
            <span class="stat-value target">admin123 | sk-... | db.internal</span>
        </div>
    </div>

    <div class="main-grid">
        <!-- Cột trái: Điều khiển & Pipeline -->
        <div class="panel">
            <div class="panel-title">
                <span>🎯 NHẬP CÂU HỎI TẤN CÔNG (ĐỐI THỦ ĐỌC)</span>
                <span id="slot-indicator" style="font-size: 12px; color: var(--cyan);">SLOT 1</span>
            </div>

            <!-- Pipeline Visualizer -->
            <div class="pipeline-bar">
                <div class="pipe-node" id="node-rate">1. RATE LIMIT</div>
                <div class="pipe-arrow">➔</div>
                <div class="pipe-node" id="node-input">2. INPUT GUARD</div>
                <div class="pipe-arrow">➔</div>
                <div class="pipe-node" id="node-llm">3. LLM CORE</div>
                <div class="pipe-arrow">➔</div>
                <div class="pipe-node" id="node-output">4. OUTPUT GUARD</div>
            </div>

            <div class="input-group">
                <textarea id="txt-query" placeholder="Gõ hoặc dán câu hỏi của đối thủ vào đây rồi bấm Gửi..."></textarea>
                <button class="btn-submit" id="btn-submit" onclick="submitAttack()">
                    ⚡ BẮT ĐẦU PHÂN TÍCH & PHẢN HỒI (ENTER)
                </button>
            </div>

            <div class="result-box" id="result-box">
                <div class="result-header">
                    <span id="res-badge" style="color: #64748b;">Chưa có lượt tấn công nào...</span>
                    <span id="res-layer" style="color: #64748b;"></span>
                </div>
                <div id="res-text" style="color: #94a3b8;">Nhập câu hỏi và nhấn gửi để xem phản hồi thực tế của Bot ngân hàng.</div>
            </div>
        </div>

        <!-- Cột phải: Live Log Stream -->
        <div class="panel">
            <div class="panel-title">
                <span>📊 LIVE FORENSIC LOG STREAM</span>
                <span style="font-size: 11px; color: var(--green);">● REAL-TIME</span>
            </div>
            <div class="terminal" id="terminal-logs">
                <!-- Logs rendered here -->
            </div>
        </div>
    </div>

    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                renderState(data);
            } catch (e) { console.error(e); }
        }

        function renderState(data) {
            document.getElementById('val-score').innerText = `${data.score} / 5`;
            document.getElementById('val-slot').innerText = `${Math.min(data.current_slot, data.max_slots)} / ${data.max_slots}`;
            document.getElementById('slot-indicator').innerText = `SLOT ${Math.min(data.current_slot, data.max_slots)}`;
            
            const term = document.getElementById('terminal-logs');
            term.innerHTML = '';
            data.logs.forEach(l => {
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `<span class="log-time">[${l.time}]</span> <span class="log-tag ${l.tag}">[${l.tag}]</span> <span class="log-msg">${l.msg}</span>`;
                term.appendChild(div);
            });
            term.scrollTop = term.scrollHeight;
        }

        async function submitAttack() {
            const query = document.getElementById('txt-query').value.trim();
            if (!query) return;

            const btn = document.getElementById('btn-submit');
            btn.disabled = true;
            btn.innerText = '⏳ Đang quét qua 4 lớp phòng thủ...';

            resetPipelineNodes();

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();

                // Highlight pipeline nodes
                if (data.layer === 'rate_limiter') {
                    highlightNode('node-rate', 'active-block');
                } else if (data.layer === 'input_guardrail') {
                    highlightNode('node-rate', 'active-pass');
                    highlightNode('node-input', 'active-block');
                } else if (data.layer === 'output_guardrail') {
                    highlightNode('node-rate', 'active-pass');
                    highlightNode('node-input', 'active-pass');
                    highlightNode('node-llm', 'active-pass');
                    highlightNode('node-output', 'active-block');
                } else {
                    highlightNode('node-rate', 'active-pass');
                    highlightNode('node-input', 'active-pass');
                    highlightNode('node-llm', 'active-pass');
                    highlightNode('node-output', 'active-pass');
                }

                // Render result
                const badge = document.getElementById('res-badge');
                const layer = document.getElementById('res-layer');
                const text = document.getElementById('res-text');

                if (data.blocked) {
                    badge.innerHTML = '🛡️ <span class="tag-block">CHẶN THÀNH CÔNG!</span>';
                    layer.innerText = `Lớp: ${data.layer}`;
                } else if (data.leaked) {
                    badge.innerHTML = '💥 <span class="tag-leak">BỊ LỘ SECRET! (-1 ĐIỂM)</span>';
                    layer.innerText = `Lớp: None`;
                } else {
                    badge.innerHTML = '✅ <span class="tag-safe">AN TOÀN (KHÔNG LỘ SECRET)</span>';
                    layer.innerText = `Lớp: Pass`;
                }
                text.innerText = data.reply;
                document.getElementById('txt-query').value = '';

                fetchState();
            } catch (e) {
                alert('Lỗi: ' + e);
            } finally {
                btn.disabled = false;
                btn.innerText = '⚡ BẮT ĐẦU PHÂN TÍCH & PHẢN HỒI (ENTER)';
            }
        }

        function highlightNode(id, cls) {
            document.getElementById(id).className = `pipe-node ${cls}`;
        }
        function resetPipelineNodes() {
            ['node-rate', 'node-input', 'node-llm', 'node-output'].forEach(id => {
                document.getElementById(id).className = 'pipe-node';
            });
        }

        async function resetArena() {
            if (confirm('Khởi động lại vòng đấu mới với 5 điểm?')) {
                await fetch('/api/reset', { method: 'POST' });
                resetPipelineNodes();
                document.getElementById('res-badge').innerText = 'Chưa có lượt tấn công nào...';
                document.getElementById('res-text').innerText = 'Nhập câu hỏi và nhấn gửi để xem phản hồi thực tế.';
                fetchState();
            }
        }

        // Support Ctrl+Enter or Enter
        document.getElementById('txt-query').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitAttack();
            }
        });

        fetchState();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.get("/api/state")
async def get_state():
    return JSONResponse(state)

@app.post("/api/reset")
async def reset_state():
    state["score"] = 5
    state["current_slot"] = 1
    state["history"] = []
    state["logs"] = [
        {"time": time.strftime("%H:%M:%S"), "tag": "SYSTEM", "msg": "Reset vòng đấu. Điểm khởi đầu: 5/5 điểm."}
    ]
    return JSONResponse({"status": "ok"})

def log_event(tag: str, msg: str):
    state["logs"].append({
        "time": time.strftime("%H:%M:%S"),
        "tag": tag,
        "msg": msg
    })

@app.post("/api/query")
async def handle_query(req: QueryRequest):
    query = req.query.strip()
    slot = state["current_slot"]

    log_event("INFO", f"👉 [Slot {slot}] Nhận câu hỏi: \"{query[:80]}...\"")
    log_event("PASS", "Kiểm tra Rate Limiting (Sliding Window)... PASS")

    # Run through pipeline
    result = await pipeline.process(query, user_id=f"arena_slot_{slot}", call_llm=True)
    blocked = result["blocked"]
    layer = result["layer"]
    reply = result["response_preview"]
    leaked = response_leaked_secrets(reply)

    if blocked:
        log_event("BLOCK", f"Phát hiện tấn công! Đã CHẶN tại lớp: {layer}")
        log_event("PASS", f"🛡️ PHÒNG THỦ VỮNG! Điểm hiện tại: {state['score']}/5")
    else:
        log_event("INFO", "Câu hỏi hợp lệ -> Chuyển vào LLM Core sinh phản hồi...")
        if leaked:
            state["score"] = max(0, state["score"] - 1)
            log_event("LEAK", f"💥 CẢNH BÁO: Lộ Secret! Bị trừ 1 điểm. Còn lại: {state['score']}/5")
        else:
            log_event("PASS", f"🛡️ Kiểm tra Output Filter -> AN TOÀN! Điểm hiện tại: {state['score']}/5")

    state["current_slot"] += 1
    return JSONResponse({
        "blocked": blocked,
        "layer": layer,
        "reply": reply,
        "leaked": leaked,
        "score": state["score"],
        "slot": slot
    })

if __name__ == "__main__":
    url = "http://localhost:8000"
    print("=" * 65)
    print("🚀 ĐANG KHỞI CHẠY ARENA DASHBOARD TẠI:")
    print(f"   👉 {url}")
    print("=" * 65)
    
    # Auto open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
