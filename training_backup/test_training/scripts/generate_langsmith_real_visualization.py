#!/usr/bin/env python3
"""Generate a redacted HTML visualization from real LangSmith raw parquet rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "langsmith_pipeline_visualization.html"
PARQUETS = [
    Path("/tmp/langsmith_raw_sample.parquet"),
    Path("/tmp/langsmith_raw_sample_large.parquet"),
    Path("/tmp/langsmith_raw_sample_2.parquet"),
]

SELECTED_THREADS = [
    "065f3f48-3bf1-4c19-b363-0614ba7418d0",
    "72abdc70-3ca6-4fb9-93fe-bb88b63469df",
    "1fd44340-deaf-4d0b-9b5c-866bd0d2871e",
]

SOURCE_URIS = [
    "s3://verdent-external-public/langsmith_exports/export_id=02235b1c-5dad-4348-9145-dac711388030/tenant_id=b7e99591-5365-4a94-87da-822c93245ac3/session_id=d151628e-5260-43c5-af46-4ce710778a2b/resource=runs/year=2026/month=3/day=14/1bd7eaf6-ce04-4209-b0a8-f7c376ff2c4a.parquet",
    "s3://verdent-external-public/langsmith_exports/export_id=02235b1c-5dad-4348-9145-dac711388030/tenant_id=b7e99591-5365-4a94-87da-822c93245ac3/session_id=d151628e-5260-43c5-af46-4ce710778a2b/resource=runs/year=2026/month=3/day=14/9c024fb6-df6d-499f-ae1f-9256aa03924a.parquet",
    "s3://verdent-external-public/langsmith_exports/export_id=02bf9b81-e2a4-45ac-96df-d4da0a97601a/tenant_id=b7e99591-5365-4a94-87da-822c93245ac3/session_id=d151628e-5260-43c5-af46-4ce710778a2b/resource=runs/year=2026/month=4/day=10/a77b6f03-7601-49b8-a291-b1d92224f528.parquet",
]

MARKERS = [
    "compact_summary",
    "persisted_summary",
    "persisted-output",
    "<persisted-output>",
    "compression",
    "context compression",
    "micro_compact",
    "thinking",
]

SIDECAR_SIGNATURES = [
    "You validate Unix-shell commands",
    "You validate Windows cmd.exe commands",
    "You validate PowerShell commands",
    "You are a task routing analyzer",
    "You are a task analyzer. Based on the conversation history",
    "You are an intelligent assistant monitoring the conversation state",
    "You decide whether to recommend Verdent marketplace integrations",
    "You are a suggestion assistant",
    "You are a lesson summarizer",
    "You are a coding assistant focused on context compression",
    "You are a context compression specialist",
    "You are a session context compressor",
]


def loads(value: Any) -> Any:
    if value is None or value == "":
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def dumps(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def sha(value: Any, n: int = 12) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()[:n]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    extra = loads(row.get("extra"))
    return extra.get("metadata", {}) if isinstance(extra, dict) else {}


def thread_id(row: dict[str, Any]) -> str:
    md = metadata(row)
    return str(md.get("thread_id") or md.get("session_id") or "")


def model(row: dict[str, Any]) -> str:
    return str(metadata(row).get("ls_model_name") or "")


def marker_hits(*values: Any) -> list[str]:
    text = "\n".join(dumps(v) for v in values).lower()
    return [m for m in MARKERS if m.lower() in text]


def content_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return ["text"]
    if isinstance(content, list):
        blocks = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(str(item.get("type") or "dict"))
            else:
                blocks.append(type(item).__name__)
        return blocks
    if content is None:
        return []
    return [type(content).__name__]


def summarize_message(msg: Any) -> dict[str, Any]:
    if not isinstance(msg, dict):
        raw = dumps(msg)
        return {
            "role": "unknown",
            "content_len": len(raw),
            "content_hash": sha(raw),
            "block_types": [type(msg).__name__],
            "markers": marker_hits(raw),
            "signature": sha({"role": "unknown", "content": raw}, 16),
        }
    content = msg.get("content")
    normalized = {"role": msg.get("role", ""), "content": content}
    return {
        "role": str(msg.get("role") or ""),
        "content_len": len(dumps(content)),
        "content_hash": sha(content),
        "block_types": content_blocks(content),
        "markers": marker_hits(content),
        "tool_call_count": len(msg.get("tool_calls") or []),
        "signature": sha(normalized, 16),
    }


def summarize_payload(value: Any) -> dict[str, Any]:
    payload = loads(value)
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        messages = []
    tools = payload.get("tools") if isinstance(payload, dict) else None
    return {
        "hash": sha(payload),
        "bytes": len(dumps(value)),
        "keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "message_count": len(messages),
        "messages": [summarize_message(m) for m in messages],
        "tools_count": len(tools) if isinstance(tools, list) else 0,
        "markers": marker_hits(payload),
    }


def is_sidecar(row: dict[str, Any]) -> bool:
    messages = summarize_payload(row.get("inputs")).get("messages", [])
    if not messages:
        return False
    raw = loads(row.get("inputs"))
    raw_msgs = raw.get("messages") if isinstance(raw, dict) else []
    if not raw_msgs or not isinstance(raw_msgs[0], dict):
        return False
    content = dumps(raw_msgs[0].get("content"))
    return any(sig in content for sig in SIDECAR_SIGNATURES)


def safe_row(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    md = metadata(row)
    tags = loads(row.get("tags"))
    if not isinstance(tags, list):
        tags = []
    return {
        "id": str(row.get("id") or ""),
        "trace_id": str(row.get("trace_id") or ""),
        "thread_id": thread_id(row),
        "project_session_id": str(row.get("session_id") or ""),
        "parent_run_id": str(row.get("parent_run_id") or ""),
        "dotted_order": str(row.get("dotted_order") or ""),
        "name": str(row.get("name") or ""),
        "run_type": str(row.get("run_type") or ""),
        "is_root": bool(row.get("is_root")),
        "status": str(row.get("status") or ""),
        "start_time": str(row.get("start_time") or ""),
        "end_time": str(row.get("end_time") or ""),
        "model": model(row),
        "provider": str(md.get("ls_provider") or ""),
        "metadata_keys": sorted(md.keys()),
        "tags": [str(t) for t in tags[:12]],
        "total_tokens": row.get("total_tokens"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "input": summarize_payload(row.get("inputs")),
        "output": summarize_payload(row.get("outputs")),
        "extra_hash": sha(loads(row.get("extra"))),
        "row_markers": marker_hits(row.get("inputs"), row.get("outputs"), row.get("extra")),
        "sidecar": is_sidecar(row),
        "source_file": source_file,
    }


def read_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in PARQUETS:
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            rid = str(row.get("id") or "")
            if rid in seen:
                continue
            seen.add(rid)
            all_rows.append(safe_row(row, path.name))

    selected = [r for r in all_rows if r["thread_id"] in SELECTED_THREADS]
    selected.sort(key=lambda r: (r["thread_id"], r["start_time"], r["dotted_order"], r["id"]))
    return all_rows, selected


def summarize_threads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["thread_id"]].append(row)
    out = []
    for tid, items in grouped.items():
        starts = [r["start_time"] for r in items if r["start_time"]]
        out.append(
            {
                "thread_id": tid,
                "rows": len(items),
                "traces": len({r["trace_id"] for r in items}),
                "llm_runs": sum(r["run_type"] == "llm" for r in items),
                "tool_runs": sum(r["run_type"] == "tool" for r in items),
                "chain_runs": sum(r["run_type"] == "chain" for r in items),
                "models": dict(Counter(r["model"] for r in items if r["model"])),
                "markers": dict(Counter(m for r in items for m in r["row_markers"])),
                "start": min(starts) if starts else "",
                "end": max(starts) if starts else "",
            }
        )
    out.sort(key=lambda x: (x["thread_id"] not in SELECTED_THREADS, x["thread_id"]))
    return out


def scan_summary(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_files": [p.name for p in PARQUETS],
        "scanned_rows": len(rows),
        "scanned_threads": len({r["thread_id"] for r in rows}),
        "scanned_traces": len({r["trace_id"] for r in rows}),
        "selected_rows": len(selected_rows),
        "selected_threads": len({r["thread_id"] for r in selected_rows}),
        "selected_traces": len({r["trace_id"] for r in selected_rows}),
    }


def build_html(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangSmith 真实 parquet row 聚合可视化</title>
  <style>
    :root {{
      --bg:#f6f7f9; --ink:#172033; --muted:#667085; --line:#d8dee8; --panel:#fff;
      --soft:#eef2f7; --green:#147d55; --green-soft:#e7f5ee; --red:#b42318;
      --red-soft:#fdecec; --blue:#245dc1; --blue-soft:#e8f0ff; --amber:#946200;
      --amber-soft:#fff4d7; --violet:#6741a2; --violet-soft:#f0e9ff;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); line-height:1.45; }}
    header {{ position:sticky; top:0; z-index:20; background:rgba(246,247,249,.94); backdrop-filter:blur(12px); border-bottom:1px solid var(--line); }}
    .bar {{ max-width:1480px; margin:0 auto; padding:14px 20px; display:grid; grid-template-columns:minmax(280px,1fr) auto; gap:16px; align-items:center; }}
    h1 {{ margin:0; font-size:20px; line-height:1.2; letter-spacing:0; }}
    .subtitle {{ margin-top:4px; color:var(--muted); font-size:13px; }}
    .controls {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
    label {{ color:var(--muted); font-size:13px; }}
    select,button {{ height:36px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); font:inherit; padding:0 10px; }}
    button {{ cursor:pointer; min-width:86px; }}
    button.active {{ border-color:var(--blue); background:var(--blue-soft); color:var(--blue); font-weight:650; }}
    main {{ max-width:1480px; margin:0 auto; padding:18px 20px 36px; }}
    .notice,.warning {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px 14px; margin-bottom:14px; color:#344054; font-size:14px; }}
    .warning {{ border-left:4px solid var(--amber); background:var(--amber-soft); color:#533b00; }}
    .warning.good {{ border-left-color:var(--green); background:var(--green-soft); color:#14563e; }}
    .warning.bad {{ border-left-color:var(--red); background:var(--red-soft); color:#6d201b; }}
    .metrics {{ display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
    .metric .value {{ font-size:24px; font-weight:750; }}
    .metric .label {{ color:var(--muted); font-size:12px; margin-top:2px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }}
    .full {{ grid-column:1 / -1; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .panel h2 {{ margin:0; padding:12px 14px; font-size:15px; border-bottom:1px solid var(--line); background:#fbfcfe; letter-spacing:0; }}
    .panel-body {{ padding:14px; }}
    .table-wrap {{ overflow:auto; max-height:520px; border:1px solid var(--line); border-radius:6px; }}
    table {{ width:100%; border-collapse:collapse; min-width:1180px; font-size:12px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }}
    th {{ position:sticky; top:0; background:#f9fafb; z-index:1; color:#344054; font-weight:700; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:12px; background:var(--soft); padding:1px 4px; border-radius:4px; }}
    .flow {{ display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); gap:8px; margin:4px 0 14px; }}
    .flow-step {{ min-height:78px; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
    .flow-step strong {{ display:block; margin-bottom:4px; font-size:13px; }}
    .flow-step span {{ display:block; color:var(--muted); font-size:12px; }}
    .tag {{ display:inline-flex; align-items:center; min-height:22px; border-radius:999px; padding:2px 8px; font-size:12px; color:#344054; background:var(--soft); border:1px solid transparent; margin:1px 2px 1px 0; }}
    .tag.blue {{ color:var(--blue); background:var(--blue-soft); }}
    .tag.green {{ color:var(--green); background:var(--green-soft); }}
    .tag.red {{ color:var(--red); background:var(--red-soft); }}
    .tag.amber {{ color:var(--amber); background:var(--amber-soft); }}
    .tag.violet {{ color:var(--violet); background:var(--violet-soft); }}
    .thread-stack {{ display:grid; gap:12px; }}
    .thread-card,.sample {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }}
    .thread-head,.sample-head {{ padding:10px 12px; background:#fbfcfe; border-bottom:1px solid var(--line); display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
    .thread-title,.sample-title {{ font-weight:750; margin-right:4px; }}
    .trace-grid {{ display:grid; gap:10px; padding:12px; }}
    .messages {{ display:grid; gap:7px; padding:10px; }}
    .msg {{ display:grid; grid-template-columns:112px 1fr; gap:8px; border:1px solid var(--line); border-radius:6px; padding:8px; background:#fff; }}
    .msg.train {{ border-color:#9bd6bd; background:var(--green-soft); }}
    .msg.mask {{ border-color:#f3b5b1; background:var(--red-soft); }}
    .msg.context {{ background:#f9fafb; }}
    .role {{ font-weight:750; color:#344054; }}
    .content {{ white-space:pre-wrap; word-break:break-word; font-size:13px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:7px; }}
    .hidden {{ display:none; }}
    @media (max-width:980px) {{
      .bar {{ grid-template-columns:1fr; }} .controls {{ justify-content:flex-start; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid {{ grid-template-columns:1fr; }}
      .flow {{ grid-template-columns:1fr; }} .msg {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>真实 LangSmith raw parquet row → 训练样本：trace-first vs thread-first RichMessage</h1>
        <div class="subtitle">生成时间：{generated_at}。所有消息内容已去敏：仅保留 role、长度、hash、block 类型和 compact marker。</div>
      </div>
      <div class="controls">
        <label for="targetModel">目标训练模型</label>
        <select id="targetModel"></select>
        <button class="active" data-view="compare">对比</button>
        <button data-view="raw">Raw</button>
        <button data-view="current">现流程</button>
        <button data-view="rich">Rich</button>
        <button data-view="steps">Thread步骤</button>
        <button data-view="docs">导出说明</button>
      </div>
    </div>
  </header>
  <main>
    <div class="notice">
      这页来自真实 S3 raw parquet。物理行单位是 <code>run</code>；<code>trace_id</code> 和 <code>thread_id/session_id</code> 都是后续聚合键。本页选取 3 个 thread，并嵌入这些 thread 在已下载 parquet 文件中的全部 run rows。
    </div>
    <section class="metrics" id="metrics"></section>
    <section class="panel full" id="section-flow">
      <h2>处理路径</h2>
      <div class="panel-body">
        <div class="flow">
          <div class="flow-step"><strong>Raw parquet row</strong><span>一行是一个 run/span，带 trace_id、parent_run_id、dotted_order、inputs/outputs。</span></div>
          <div class="flow-step"><strong>当前 trace-first</strong><span>先 group by trace_id，trace 内取最后一个 main LLM run 构候选样本。</span></div>
          <div class="flow-step"><strong>Flatten</strong><span>变成裸 Message 后 source_model/source_run_id 不再是训练格式字段。</span></div>
          <div class="flow-step"><strong>Thread dedup</strong><span>再按 session_id/thread_id 做 prefix/turn dedup，只能依赖内容签名。</span></div>
          <div class="flow-step"><strong>Rich thread-first</strong><span>先按 thread 聚合 raw runs，保留 source metadata 到 mask 阶段。</span></div>
        </div>
        <span class="tag green">train</span>
        <span class="tag red">loss=0 / masked</span>
        <span class="tag amber">旧流程 metadata 不足</span>
        <span class="tag violet">source metadata retained</span>
      </div>
    </section>
    <section class="panel full" id="section-raw">
      <h2>选中 3 个 thread 的全部 raw run rows（去敏摘要）</h2>
      <div class="panel-body">
        <div class="table-wrap"><table id="rawTable"></table></div>
      </div>
    </section>
    <section class="panel full" id="section-steps">
      <h2>thread-first RichMessage 分步骤处理</h2>
      <div class="panel-body">
        <div class="warning good">这里展示从 run 级候选空间收敛到最终训练窗口的过程。关系判断是基于去敏 hash/marker 的可视化启发式，生产实现应继续用 parent_run_id、dotted_order、tool_call_id 和完整结构化 block 做严谨判定。</div>
        <div id="stepsView" class="thread-stack"></div>
      </div>
    </section>
    <div class="grid">
      <section class="panel" id="section-current">
        <h2>当前 trace-first 样本构建</h2>
        <div class="panel-body">
          <div class="warning bad">这里展示当前抽象：trace 内最后一个 main LLM run 的 inputs + outputs 形成候选样本。因为训练 JSONL 不保存 source_model/source_run_id，按目标模型 mask 只能做事后推断。</div>
          <div id="currentView" class="thread-stack"></div>
        </div>
      </section>
      <section class="panel" id="section-rich">
        <h2>thread-first RichMessage 样本构建</h2>
        <div class="panel-body">
          <div class="warning good">RichMessage 在 mask 阶段仍保留 source_run_id、source_trace_id、source_model、window_id，可表达热切模型、compact history、重复 turn 等规则。</div>
          <div id="richView" class="thread-stack"></div>
        </div>
      </section>
    </div>
    <section class="panel full" id="section-docs">
      <h2>LangSmith 官方结构/导出说明摘记</h2>
      <div class="panel-body" id="docsView"></div>
    </section>
  </main>
  <script id="embedded-data" type="application/json">{encoded}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('embedded-data').textContent);
    const rows = DATA.rows;
    const selectedThreads = DATA.selected_threads;
    const docs = DATA.docs;

    const by = (items, fn) => items.reduce((acc, item) => {{
      const key = fn(item) || '';
      (acc[key] ||= []).push(item);
      return acc;
    }}, {{}});
    const sig = (m) => `${{m.role}}:${{m.signature}}`;
    const short = (s, n = 10) => String(s || '').slice(0, n);

    function mainLlmRuns(traceRows) {{
      return traceRows
        .filter(r => r.run_type === 'llm' && !r.sidecar)
        .sort((a, b) => String(a.dotted_order).localeCompare(String(b.dotted_order)));
    }}

    function outputIndex(threadRows) {{
      const out = new Map();
      threadRows
        .filter(r => r.run_type === 'llm' && !r.sidecar)
        .sort((a, b) => String(a.start_time).localeCompare(String(b.start_time)))
        .forEach(r => {{
          (r.output.messages || []).forEach(m => {{
            const key = sig(m);
            const val = {{ source_run_id:r.id, source_trace_id:r.trace_id, source_model:r.model, source_time:r.start_time }};
            const arr = out.get(key) || [];
            arr.push(val);
            out.set(key, arr);
          }});
        }});
      return out;
    }}

    function traceFirstSamples() {{
      const traces = by(rows, r => r.trace_id);
      return Object.entries(traces).map(([traceId, traceRows]) => {{
        const llms = mainLlmRuns(traceRows);
        const root = traceRows.find(r => r.is_root) || traceRows[0];
        if (!llms.length) return {{ traceId, thread_id:root.thread_id, skipped:true, reason:'no main LLM run' }};
        const first = llms[0], last = llms[llms.length - 1];
        const firstNonSystem = (first.input.messages || []).filter(m => m.role !== 'system').length;
        const system = (last.input.messages || []).filter((m, i) => i === 0 && m.role === 'system');
        const nonSystem = (last.input.messages || []).filter((m, i) => !(i === 0 && m.role === 'system'));
        const history = nonSystem.slice(0, firstNonSystem);
        const newer = nonSystem.slice(firstNonSystem);
        const messages = [
          ...system.map(m => ({{...m, loss:0, part:'system'}})),
          ...history.map(m => ({{...m, loss:0, part:'history_or_first_input'}})),
          ...newer.map(m => ({{...m, loss:null, part:'new_input_after_boundary'}})),
          ...(last.output.messages || []).map(m => ({{...m, loss:null, part:'last_output'}})),
        ];
        const idx = outputIndex(traceRows);
        messages.forEach(m => {{
          const hit = (idx.get(sig(m)) || [])[0];
          if (hit) Object.assign(m, hit);
        }});
        const lastInputSigs = new Set((last.input.messages || []).map(sig));
        const lastOutputSigs = new Set((last.output.messages || []).map(sig));
        const lostOutputs = [];
        llms.forEach(r => {{
          if (r.id === last.id) return;
          (r.output.messages || []).forEach(m => {{
            if (!lastInputSigs.has(sig(m)) && !lastOutputSigs.has(sig(m))) {{
              lostOutputs.push({{ run_id:r.id, model:r.model, role:m.role, hash:m.content_hash, len:m.content_len }});
            }}
          }});
        }});
        return {{
          traceId, thread_id:root.thread_id, firstRunId:first.id, lastRunId:last.id,
          firstModel:first.model, lastModel:last.model, firstNonSystem, llmCount:llms.length,
          rowCount:traceRows.length, messages, lostOutputs,
        }};
      }}).sort((a,b) => String(a.thread_id).localeCompare(String(b.thread_id)) || String(a.traceId).localeCompare(String(b.traceId)));
    }}

    function isPrefixExtension(prevRun, run) {{
      const prev = [...(prevRun.input.messages || []), ...(prevRun.output.messages || [])].map(sig);
      const curr = (run.input.messages || []).map(sig);
      if (prev.length > curr.length) return false;
      for (let i = 0; i < prev.length; i++) if (prev[i] !== curr[i]) return false;
      return true;
    }}

    function richSamples(targetModel) {{
      const threads = by(rows, r => r.thread_id);
      return Object.entries(threads).map(([threadId, threadRows]) => {{
        const samples = candidateWindowsForThread(threadId, threadRows).samples;
        samples.sort((a,b) => String(a.start_time).localeCompare(String(b.start_time)));
        applyMasks(samples, targetModel);
        return {{ threadId, rows:threadRows, samples }};
      }}).sort((a,b) => selectedThreads.indexOf(a.threadId) - selectedThreads.indexOf(b.threadId));
    }}

    function runHasCompactMarker(run) {{
      const markers = [...(run.row_markers || []), ...(run.input.markers || []), ...(run.output.markers || [])]
        .map(x => String(x).toLowerCase());
      return markers.some(x => x.includes('compact') || x.includes('compression') || x.includes('persisted'));
    }}

    function relationBetween(prevRun, run) {{
      if (!prevRun) return {{ kind:'start', label:'start', reason:'first main LLM run in this trace', overlap:0 }};
      if (isPrefixExtension(prevRun, run)) {{
        return {{ kind:'prefix_extend', label:'prefix extend', reason:'prev input+output is a strict prefix of current input', overlap:null }};
      }}
      const prevSeq = [...(prevRun.input.messages || []), ...(prevRun.output.messages || [])].map(sig);
      const currSeq = (run.input.messages || []).map(sig);
      const currSet = new Set(currSeq);
      const overlap = prevSeq.filter(x => currSet.has(x)).length;
      const prevOut = (prevRun.output.messages || []).map(sig);
      const replaysPrevOutput = prevOut.some(x => currSet.has(x));
      if (runHasCompactMarker(run)) {{
        return {{ kind:'compact_or_reset', label:'compact/reset', reason:'compact/persist/compression marker is present near this run', overlap }};
      }}
      if (replaysPrevOutput || overlap > 0) {{
        return {{ kind:'branch_or_partial_replay', label:'branch/partial replay', reason:'some previous messages reappear, but not as a full prefix', overlap }};
      }}
      return {{ kind:'fork_or_side_branch', label:'fork/side branch', reason:'no strict prefix and little/no replay overlap', overlap }};
    }}

    function candidateWindowsForThread(threadId, threadRows) {{
      const idx = outputIndex(threadRows);
      const traces = by(threadRows, r => r.trace_id);
      const samples = [];
      const transitions = [];
      Object.entries(traces).forEach(([traceId, traceRows]) => {{
        const llms = mainLlmRuns(traceRows);
        let prev = null, current = null, n = 0;
        llms.forEach(run => {{
          const rel = relationBetween(prev, run);
          if (prev) {{
            transitions.push({{
              traceId,
              prevRunId: prev.id,
              runId: run.id,
              prevModel: prev.model,
              model: run.model,
              kind: rel.kind,
              label: rel.label,
              reason: rel.reason,
              overlap: rel.overlap,
              inputMsgs: run.input.message_count,
              outputMsgs: run.output.message_count,
              markers: [...new Set([...(run.row_markers || []), ...(run.input.markers || [])])],
            }});
          }}
          const reset = !prev || rel.kind !== 'prefix_extend';
          if (reset) {{
            if (current) samples.push(buildWindow(current, idx));
            n += 1;
            current = {{ threadId, traceId, windowId:`${{short(traceId)}}/w${{n}}`, runs:[] }};
          }}
          current.runs.push(run);
          prev = run;
        }});
        if (current) samples.push(buildWindow(current, idx));
      }});
      samples.sort((a,b) => String(a.start_time).localeCompare(String(b.start_time)));
      return {{ samples, transitions }};
    }}

    function selectWindows(samples, strategy) {{
      const byTrace = by(samples, s => s.traceId);
      const selected = [];
      Object.values(byTrace).forEach(group => {{
        const sorted = [...group].sort((a,b) => String(a.start_time).localeCompare(String(b.start_time)));
        if (strategy === 'terminal') {{
          selected.push(sorted[sorted.length - 1]);
        }} else {{
          selected.push([...group].sort((a,b) => {{
            const alen = a.messages.length, blen = b.messages.length;
            if (alen !== blen) return blen - alen;
            return String(b.start_time).localeCompare(String(a.start_time));
          }})[0]);
        }}
      }});
      return selected.sort((a,b) => String(a.start_time).localeCompare(String(b.start_time)));
    }}

    function cloneSamples(samples) {{
      return JSON.parse(JSON.stringify(samples));
    }}

    function countKinds(items, key) {{
      return items.reduce((acc, item) => {{
        const k = item[key] || 'unknown';
        acc[k] = (acc[k] || 0) + 1;
        return acc;
      }}, {{}});
    }}

    function maskStats(samples) {{
      const msgs = samples.flatMap(s => s.messages || []);
      const generated = msgs.filter(m => m.role === 'assistant' || m.role === 'tool_call');
      const train = generated.filter(m => m.loss !== 0);
      const masked = generated.filter(m => m.loss === 0);
      return {{
        windows: samples.length,
        messages: msgs.length,
        generated: generated.length,
        train: train.length,
        masked: masked.length,
        reasons: countKinds(masked, 'mask_reason'),
        models: countKinds(generated.filter(m => m.source_model), 'source_model'),
      }};
    }}

    function prepareSelected(samples, strategy, targetModel) {{
      const selected = cloneSamples(selectWindows(samples, strategy));
      applyMasks(selected, targetModel);
      return selected;
    }}

    function enrichInput(m, sample, idx) {{
      const hits = idx.get(sig(m)) || [];
      const hit = hits[0];
      let source_kind = m.role === 'system' ? 'system' : 'user_or_history';
      if ((m.markers || []).some(x => x.includes('persisted') || x.includes('compact'))) source_kind = 'compact_history';
      if (hit) source_kind = 'llm_output_replayed_as_context';
      return {{...m, source_kind, source_run_id:hit?.source_run_id || '', source_trace_id:hit?.source_trace_id || '', source_model:hit?.source_model || '', window_id:sample.windowId}};
    }}

    function buildWindow(sample, idx) {{
      const last = sample.runs[sample.runs.length - 1];
      const messages = [
        ...(last.input.messages || []).map(m => enrichInput(m, sample, idx)),
        ...(last.output.messages || []).map(m => ({{...m, source_kind:'llm_output', source_run_id:last.id, source_trace_id:last.trace_id, source_model:last.model, window_id:sample.windowId}})),
      ];
      return {{
        threadId:sample.threadId, traceId:sample.traceId, windowId:sample.windowId,
        runIds:sample.runs.map(r => r.id), models:[...new Set(sample.runs.map(r => r.model).filter(Boolean))],
        start_time:sample.runs[0]?.start_time || '', messages,
      }};
    }}

    function applyMasks(samples, targetModel) {{
      const seen = new Set();
      samples.forEach(s => s.messages.forEach(m => {{
        const trainableRole = m.role === 'assistant' || m.role === 'tool_call';
        if (!trainableRole) {{ m.loss = 0; m.mask_reason = m.source_kind === 'compact_history' ? 'compact_history' : 'prompt_or_context'; return; }}
        if (targetModel !== 'all' && m.source_model && m.source_model !== targetModel) {{ m.loss = 0; m.mask_reason = 'source_model_mismatch'; return; }}
        const key = `${{s.threadId}}:${{sig(m)}}`;
        if (seen.has(key)) {{ m.loss = 0; m.mask_reason = 'thread_duplicate_turn'; return; }}
        m.loss = null; m.mask_reason = 'train'; seen.add(key);
      }}));
    }}

    function renderMetrics(current, rich) {{
      const allSamples = rich.flatMap(t => t.samples);
      const richTrain = allSamples.flatMap(s => s.messages).filter(m => (m.role === 'assistant' || m.role === 'tool_call') && m.loss !== 0).length;
      const currentTrain = current.flatMap(s => s.messages || []).filter(m => (m.role === 'assistant' || m.role === 'tool_call') && m.loss !== 0).length;
      const values = [
        ['选中 run rows', rows.length],
        ['Thread', new Set(rows.map(r => r.thread_id)).size],
        ['Trace', new Set(rows.map(r => r.trace_id)).size],
        ['当前/Rich 样本', `${{current.filter(s=>!s.skipped).length}} / ${{allSamples.length}}`],
        ['当前/Rich trainable', `${{currentTrain}} / ${{richTrain}}`],
      ];
      document.getElementById('metrics').innerHTML = values.map(([label,value]) => `<div class="metric"><div class="value">${{value}}</div><div class="label">${{label}}</div></div>`).join('');
    }}

    function renderRawTable() {{
      document.getElementById('rawTable').innerHTML = `<thead><tr>
        <th>run_id</th><th>thread_id</th><th>trace_id</th><th>type</th><th>model</th><th>root</th><th>sidecar</th><th>parent</th><th>time</th><th>inputs</th><th>outputs</th><th>markers</th><th>tokens</th>
      </tr></thead><tbody>${{rows.map(r => `<tr>
        <td><code>${{short(r.id,8)}}</code></td><td><code>${{short(r.thread_id,8)}}</code></td><td><code>${{short(r.trace_id,8)}}</code></td>
        <td>${{r.run_type}}<br><span class="tag">${{escapeHtml(r.name)}}</span></td><td>${{escapeHtml(r.model || '')}}</td>
        <td>${{r.is_root ? 'yes' : ''}}</td><td>${{r.sidecar ? '<span class="tag amber">sidecar</span>' : ''}}</td>
        <td><code>${{short(r.parent_run_id,8)}}</code></td><td>${{escapeHtml(r.start_time)}}</td>
        <td>${{payloadCell(r.input)}}</td><td>${{payloadCell(r.output)}}</td><td>${{tags(r.row_markers,'amber')}}</td>
        <td>${{r.total_tokens || ''}}</td>
      </tr>`).join('')}}</tbody>`;
    }}

    function payloadCell(p) {{
      return `<span class="tag blue">msgs=${{p.message_count}}</span><span class="tag">tools=${{p.tools_count}}</span><span class="tag">bytes=${{p.bytes}}</span><br>${{p.messages.slice(0,4).map(m => msgMini(m)).join('<br>')}}${{p.messages.length > 4 ? '<br><span class="tag">...</span>' : ''}}`;
    }}

    function msgMini(m) {{
      return `<span class="tag">${{m.role || 'unknown'}}</span> len=${{m.content_len}} hash=${{m.content_hash}} ${{tags(m.markers,'amber')}}`;
    }}

    function renderCurrent(current, targetModel) {{
      const grouped = by(current, s => s.thread_id);
      document.getElementById('currentView').innerHTML = Object.entries(grouped).map(([tid, samples]) => `<div class="thread-card">
        <div class="thread-head"><span class="thread-title">${{tid}}</span><span class="tag blue">${{samples.length}} trace groups</span></div>
        <div class="trace-grid">${{samples.map(s => renderCurrentSample(s, targetModel)).join('')}}</div>
      </div>`).join('');
    }}

    function renderCurrentSample(s, targetModel) {{
      if (s.skipped) return `<div class="sample"><div class="sample-head"><span class="sample-title">${{s.traceId}}</span><span class="tag red">${{s.reason}}</span></div></div>`;
      const mismatches = s.messages.filter(m => (m.role === 'assistant' || m.role === 'tool_call') && m.loss !== 0 && targetModel !== 'all' && m.source_model && m.source_model !== targetModel);
      return `<div class="sample">
        <div class="sample-head"><span class="sample-title">trace ${{short(s.traceId,8)}}</span><span class="tag">rows=${{s.rowCount}}</span><span class="tag">llm=${{s.llmCount}}</span><span class="tag violet">last=${{escapeHtml(s.lastModel)}}</span></div>
        ${{mismatches.length ? `<div class="warning bad">${{mismatches.length}} 条可训练输出推断来源模型 != ${{escapeHtml(targetModel)}}；flatten 后无法可靠写 mask。</div>` : ''}}
        ${{s.lostOutputs.length ? `<div class="warning bad">${{s.lostOutputs.length}} 条较早 LLM output 不在 last-run snapshot 中，可能是 trace 内 reset/compact/分支造成。</div>` : ''}}
        <div class="messages">${{s.messages.map(m => renderMsg(m, targetModel, false)).join('')}}</div>
      </div>`;
    }}

    function renderRich(rich) {{
      document.getElementById('richView').innerHTML = rich.map(t => `<div class="thread-card">
        <div class="thread-head"><span class="thread-title">${{t.threadId}}</span><span class="tag blue">${{t.samples.length}} windows</span><span class="tag">${{t.rows.length}} rows</span></div>
        <div class="trace-grid">${{t.samples.map(s => `<div class="sample">
          <div class="sample-head"><span class="sample-title">${{s.windowId}}</span><span class="tag">trace=${{short(s.traceId,8)}}</span><span class="tag violet">models=${{escapeHtml(s.models.join(', ') || 'none')}}</span><span class="tag">runs=${{s.runIds.length}}</span></div>
          <div class="messages">${{s.messages.map(m => renderMsg(m, '', true)).join('')}}</div>
        </div>`).join('')}}</div>
      </div>`).join('');
    }}

    function renderSteps(targetModel) {{
      const threads = by(rows, r => r.thread_id);
      document.getElementById('stepsView').innerHTML = Object.entries(threads)
        .sort((a,b) => selectedThreads.indexOf(a[0]) - selectedThreads.indexOf(b[0]))
        .map(([threadId, threadRows]) => renderThreadSteps(threadId, threadRows, targetModel))
        .join('');
    }}

    function renderThreadSteps(threadId, threadRows, targetModel) {{
      const mainRuns = threadRows.filter(r => r.run_type === 'llm' && !r.sidecar);
      const built = candidateWindowsForThread(threadId, threadRows);
      const relationCounts = countKinds(built.transitions, 'kind');
      const terminal = prepareSelected(built.samples, 'terminal', targetModel);
      const longest = prepareSelected(built.samples, 'longest', targetModel);
      const terminalStats = maskStats(terminal);
      const longestStats = maskStats(longest);
      const traceCount = new Set(threadRows.map(r => r.trace_id)).size;
      const compactRows = threadRows.filter(r => (r.row_markers || []).length).length;
      return `<div class="thread-card">
        <div class="thread-head">
          <span class="thread-title">${{threadId}}</span>
          <span class="tag blue">${{mainRuns.length}} main LLM runs</span>
          <span class="tag">${{built.samples.length}} candidate windows</span>
          <span class="tag">${{traceCount}} traces</span>
          <span class="tag amber">${{compactRows}} marked rows</span>
        </div>
        <div class="trace-grid">
          <div class="sample">
            <div class="sample-head"><span class="sample-title">1. 候选窗口</span><span class="tag">run 是原子事实</span><span class="tag violet">Rich metadata retained</span></div>
            <div class="panel-body">${{renderCandidateTable(built.samples)}}</div>
          </div>
          <div class="sample">
            <div class="sample-head"><span class="sample-title">2. prefix / branch / compact / fork 判定</span>${{objectTags(relationCounts)}}</div>
            <div class="panel-body">${{renderTransitionTable(built.transitions)}}</div>
          </div>
          <div class="sample">
            <div class="sample-head"><span class="sample-title">3. 选择训练窗口</span><span class="tag green">terminal</span><span class="tag blue">longest-prefix</span></div>
            <div class="panel-body">${{renderSelectionComparison(built.samples, terminal, longest, terminalStats, longestStats)}}</div>
          </div>
          <div class="sample">
            <div class="sample-head"><span class="sample-title">4. thread 内 dedup</span><span class="tag">signature-based</span></div>
            <div class="panel-body">${{renderReasonBlock('terminal', terminalStats)}}${{renderReasonBlock('longest-prefix', longestStats)}}</div>
          </div>
          <div class="sample">
            <div class="sample-head"><span class="sample-title">5. source_model mask</span><span class="tag violet">target=${{escapeHtml(targetModel)}}</span></div>
            <div class="panel-body">${{renderSelectedWindows('terminal', terminal)}}${{renderSelectedWindows('longest-prefix', longest)}}</div>
          </div>
        </div>
      </div>`;
    }}

    function renderCandidateTable(samples) {{
      const preview = samples.slice(0, 12);
      return `<div class="table-wrap" style="max-height:320px"><table><thead><tr>
        <th>window</th><th>trace</th><th>runs</th><th>models</th><th>messages</th><th>generated candidates</th><th>start</th>
      </tr></thead><tbody>${{preview.map(s => {{
        const generated = s.messages.filter(m => m.role === 'assistant' || m.role === 'tool_call').length;
        return `<tr><td><code>${{s.windowId}}</code></td><td><code>${{short(s.traceId,8)}}</code></td><td>${{s.runIds.length}}</td><td>${{escapeHtml(s.models.join(', ') || '')}}</td><td>${{s.messages.length}}</td><td>${{generated}}</td><td>${{escapeHtml(s.start_time)}}</td></tr>`;
      }}).join('')}}</tbody></table></div>
      ${{samples.length > preview.length ? `<div class="notice">仅展示前 ${{preview.length}} 个候选窗口；完整候选窗口数为 ${{samples.length}}。</div>` : ''}}`;
    }}

    function renderTransitionTable(transitions) {{
      if (!transitions.length) return `<div class="notice">该 thread 没有可比较的相邻 main LLM run。</div>`;
      const preview = transitions.slice(0, 18);
      return `<div class="table-wrap" style="max-height:360px"><table><thead><tr>
        <th>trace</th><th>prev run → run</th><th>relation</th><th>overlap</th><th>input/output msgs</th><th>model</th><th>markers</th><th>reason</th>
      </tr></thead><tbody>${{preview.map(t => `<tr>
        <td><code>${{short(t.traceId,8)}}</code></td>
        <td><code>${{short(t.prevRunId,8)}}</code> → <code>${{short(t.runId,8)}}</code></td>
        <td>${{relationTag(t.kind, t.label)}}</td>
        <td>${{t.overlap === null ? 'full' : t.overlap}}</td>
        <td>${{t.inputMsgs}} / ${{t.outputMsgs}}</td>
        <td>${{escapeHtml(t.model || '')}}</td>
        <td>${{tags(t.markers, 'amber')}}</td>
        <td>${{escapeHtml(t.reason)}}</td>
      </tr>`).join('')}}</tbody></table></div>
      ${{transitions.length > preview.length ? `<div class="notice">仅展示前 ${{preview.length}} 条关系；完整 transition 数为 ${{transitions.length}}。</div>` : ''}}`;
    }}

    function renderSelectionComparison(candidates, terminal, longest, terminalStats, longestStats) {{
      const rows = [
        ['terminal per trace', terminal, terminalStats, '每个 trace 保留时间上最后的 Rich window；更接近“最终状态/terminal action”。'],
        ['longest-prefix per trace', longest, longestStats, '每个 trace 保留消息数最长的 Rich window；更接近“最长上下文 snapshot”。'],
      ];
      return `<div class="notice">候选窗口不会直接全量训练；selection stage 负责从 ${{candidates.length}} 个候选里挑训练窗口。</div>
      <div class="table-wrap" style="max-height:300px"><table><thead><tr>
        <th>strategy</th><th>selected / candidate</th><th>selected windows</th><th>train / masked generated</th><th>说明</th>
      </tr></thead><tbody>${{rows.map(([name, selected, stats, note]) => `<tr>
        <td>${{name}}</td>
        <td>${{selected.length}} / ${{candidates.length}}</td>
        <td>${{selected.map(s => `<code>${{s.windowId}}</code>`).join(' ')}}</td>
        <td><span class="tag green">${{stats.train}} train</span><span class="tag red">${{stats.masked}} masked</span></td>
        <td>${{escapeHtml(note)}}</td>
      </tr>`).join('')}}</tbody></table></div>`;
    }}

    function renderReasonBlock(name, stats) {{
      return `<div class="notice"><strong>${{name}}</strong>：windows=${{stats.windows}}，messages=${{stats.messages}}，generated=${{stats.generated}}，
        <span class="tag green">train=${{stats.train}}</span><span class="tag red">masked=${{stats.masked}}</span>
        <br>mask reasons: ${{objectTags(stats.reasons)}}<br>source models: ${{objectTags(stats.models, 'violet')}}</div>`;
    }}

    function renderSelectedWindows(name, samples) {{
      return `<div class="notice"><strong>${{name}}</strong></div>
      <div class="table-wrap" style="max-height:320px"><table><thead><tr>
        <th>window</th><th>trace</th><th>models</th><th>trainable generated</th><th>masked generated</th><th>mask reasons</th>
      </tr></thead><tbody>${{samples.map(s => {{
        const generated = s.messages.filter(m => m.role === 'assistant' || m.role === 'tool_call');
        const train = generated.filter(m => m.loss !== 0).length;
        const masked = generated.filter(m => m.loss === 0);
        return `<tr><td><code>${{s.windowId}}</code></td><td><code>${{short(s.traceId,8)}}</code></td><td>${{escapeHtml(s.models.join(', ') || '')}}</td><td>${{train}}</td><td>${{masked.length}}</td><td>${{objectTags(countKinds(masked, 'mask_reason'))}}</td></tr>`;
      }}).join('')}}</tbody></table></div>`;
    }}

    function relationTag(kind, label) {{
      const color = kind === 'prefix_extend' ? 'green' : (kind === 'compact_or_reset' ? 'amber' : (kind === 'branch_or_partial_replay' ? 'blue' : 'red'));
      return `<span class="tag ${{color}}">${{escapeHtml(label)}}</span>`;
    }}

    function renderMsg(m, targetModel, rich) {{
      const train = (m.role === 'assistant' || m.role === 'tool_call') && m.loss !== 0;
      const mismatch = !rich && targetModel !== 'all' && train && m.source_model && m.source_model !== targetModel;
      const cls = train ? (mismatch ? 'mask' : 'train') : 'context';
      return `<div class="msg ${{cls}}">
        <div class="role">${{m.role || 'unknown'}}</div>
        <div><div class="content">redacted content: len=${{m.content_len}}, hash=${{m.content_hash}}, blocks=${{(m.block_types||[]).join('/')}}</div>
          <div class="meta">
            <span class="tag ${{train ? 'green' : 'red'}}">${{train ? 'train' : 'loss=0/context'}}</span>
            ${{m.part ? `<span class="tag">${{m.part}}</span>` : ''}}
            ${{m.mask_reason ? `<span class="tag">${{m.mask_reason}}</span>` : ''}}
            ${{m.source_kind ? `<span class="tag blue">${{m.source_kind}}</span>` : ''}}
            ${{m.source_model ? `<span class="tag violet">model=${{escapeHtml(m.source_model)}}</span>` : ''}}
            ${{m.source_run_id ? `<span class="tag">run=${{short(m.source_run_id,8)}}</span>` : ''}}
            ${{tags(m.markers,'amber')}}
            ${{mismatch ? `<span class="tag red">should mask</span>` : ''}}
          </div></div>
      </div>`;
    }}

    function renderDocs() {{
      document.getElementById('docsView').innerHTML = `
        <div class="warning good">官方概念文档：Project 包含多个 traces；trace 是 single operation 的 runs 集合；thread 是由共享 session_id/thread_id/conversation_id 连接起来的一组 traces。</div>
        <ul>
          <li><a href="${{docs.observability_url}}">LangSmith Observability concepts</a></li>
          <li><a href="${{docs.sdk_client_url}}">LangSmith Python SDK Client reference</a>：SDK 核心接口包括 read_run/list_runs，list_runs 返回 Run iterator，并支持 trace_id、trace_filter、tree_filter、is_root、select 等 run 查询维度。</li>
          <li><a href="${{docs.dataset_url}}">Manage datasets</a>：数据集可以从 UI 导出 CSV、JSONL 或 OpenAI fine-tuning format；这是 dataset export，不是 raw trace/thread conversation 批量导出。</li>
        </ul>
        <p>本页实际 S3 路径也印证 raw bulk export 的 <code>resource=runs</code> 目录是一张 run 级表。结构化 thread/trace conversation 仍需在下游用 <code>trace_id</code>、<code>parent_run_id</code>、<code>dotted_order</code> 和 metadata 重新构建。</p>
        <p>Raw source files:</p>
        <ul>${{DATA.source_uris.map(u => `<li><code>${{u}}</code></li>`).join('')}}</ul>
      `;
    }}

    function tags(items, color='') {{
      return (items || []).map(x => `<span class="tag ${{color}}">${{escapeHtml(x)}}</span>`).join('');
    }}
    function objectTags(obj, color='') {{
      const entries = Object.entries(obj || {{}}).filter(([,v]) => v);
      if (!entries.length) return '<span class="tag">none</span>';
      return entries.map(([k,v]) => `<span class="tag ${{color}}">${{escapeHtml(k)}}=${{v}}</span>`).join('');
    }}
    function escapeHtml(s) {{
      return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }}
    function setupModels() {{
      const models = [...new Set(rows.map(r => r.model).filter(Boolean))].sort();
      const sel = document.getElementById('targetModel');
      sel.innerHTML = [`<option value="all">不按模型 mask</option>`, ...models.map(m => `<option value="${{escapeHtml(m)}}">${{escapeHtml(m)}}</option>`)].join('');
      const opus = models.find(m => m.includes('opus')) || models[0] || 'all';
      sel.value = opus;
    }}
    function render() {{
      const target = document.getElementById('targetModel').value;
      const current = traceFirstSamples();
      const rich = richSamples(target);
      renderMetrics(current, rich);
      renderRawTable();
      renderCurrent(current, target);
      renderRich(rich);
      renderSteps(target);
      renderDocs();
    }}
    function applyView(view) {{
      ['flow','raw','current','rich','steps','docs'].forEach(name => document.getElementById(`section-${{name}}`)?.classList.add('hidden'));
      if (view === 'compare') ['flow','raw','current','rich','steps','docs'].forEach(name => document.getElementById(`section-${{name}}`)?.classList.remove('hidden'));
      else document.getElementById(`section-${{view}}`)?.classList.remove('hidden');
      document.querySelectorAll('button[data-view]').forEach(btn => btn.classList.toggle('active', btn.dataset.view === view));
    }}
    setupModels();
    document.getElementById('targetModel').addEventListener('change', render);
    document.querySelectorAll('button[data-view]').forEach(btn => btn.addEventListener('click', () => applyView(btn.dataset.view)));
    render();
    applyView('compare');
  </script>
</body>
</html>
"""


def main() -> None:
    all_rows, selected_rows = read_rows()
    data = {
        "rows": selected_rows,
        "selected_threads": SELECTED_THREADS,
        "scan_summary": scan_summary(all_rows, selected_rows),
        "thread_summary_selected": summarize_threads(selected_rows),
        "source_uris": SOURCE_URIS,
        "docs": {
            "observability_url": "https://docs.langchain.com/langsmith/observability-concepts",
            "sdk_client_url": "https://reference.langchain.com/python/langsmith/observability/sdk/client/",
            "dataset_url": "https://docs.langchain.com/langsmith/manage-datasets",
        },
    }
    OUT.write_text(build_html(data), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"selected rows: {len(selected_rows)}")
    for item in data["thread_summary_selected"]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
