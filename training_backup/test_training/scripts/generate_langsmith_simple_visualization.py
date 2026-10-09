#!/usr/bin/env python3
"""Generate a concise, result-first visualization from redacted real parquet rows."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_langsmith_real_visualization import read_rows  # noqa: E402


OUT = ROOT / "langsmith_pipeline_visualization_simple.html"
TARGET_DEFAULT = "claude-opus-4-6"


def by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get(key) or "")].append(item)
    return grouped


def sig(msg: dict[str, Any]) -> str:
    return f"{msg.get('role', '')}:{msg.get('signature', '')}"


def is_generated(msg: dict[str, Any]) -> bool:
    return msg.get("role") in {"assistant", "tool_call"}


def main_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [r for r in rows if r["run_type"] == "llm" and not r.get("sidecar")],
        key=lambda r: (r.get("dotted_order") or "", r.get("start_time") or "", r.get("id") or ""),
    )


def output_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in sorted(main_runs(rows), key=lambda r: r.get("start_time") or ""):
        for msg in run["output"].get("messages", []):
            idx[sig(msg)].append(
                {
                    "source_run_id": run["id"],
                    "source_trace_id": run["trace_id"],
                    "source_model": run["model"],
                    "source_time": run["start_time"],
                }
            )
    return idx


def trace_first_samples(thread_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = output_index(thread_rows)
    samples: list[dict[str, Any]] = []
    for trace_id, trace_rows in by(thread_rows, "trace_id").items():
        llms = main_runs(trace_rows)
        if not llms:
            continue
        first = llms[0]
        last = llms[-1]
        first_non_system = sum(1 for m in first["input"].get("messages", []) if m.get("role") != "system")
        last_input = last["input"].get("messages", [])
        system = [m for i, m in enumerate(last_input) if i == 0 and m.get("role") == "system"]
        non_system = [m for i, m in enumerate(last_input) if not (i == 0 and m.get("role") == "system")]
        history = non_system[:first_non_system]
        newer = non_system[first_non_system:]
        messages: list[dict[str, Any]] = []
        for part, loss, items in (
            ("system", 0, system),
            ("history_or_first_input", 0, history),
            ("new_input_after_boundary", None, newer),
            ("last_output", None, last["output"].get("messages", [])),
        ):
            for msg in items:
                m = dict(msg)
                m["part"] = part
                m["loss"] = loss
                hit = (idx.get(sig(m)) or [None])[0]
                if hit:
                    m.update(hit)
                elif part == "last_output":
                    m["source_run_id"] = last["id"]
                    m["source_trace_id"] = last["trace_id"]
                    m["source_model"] = last["model"]
                messages.append(m)

        last_input_sigs = {sig(m) for m in last_input}
        last_output_sigs = {sig(m) for m in last["output"].get("messages", [])}
        lost_outputs = []
        for run in llms[:-1]:
            for msg in run["output"].get("messages", []):
                if sig(msg) not in last_input_sigs and sig(msg) not in last_output_sigs:
                    lost_outputs.append({"run_id": run["id"], "model": run["model"], "hash": msg["content_hash"]})

        generated = [m for m in messages if is_generated(m)]
        samples.append(
            {
                "trace_id": trace_id,
                "first_run_id": first["id"],
                "last_run_id": last["id"],
                "first_model": first["model"],
                "last_model": last["model"],
                "main_runs": len(llms),
                "messages": len(messages),
                "context_messages": sum(1 for m in messages if not is_generated(m) or m.get("loss") == 0),
                "trainable_generated": sum(1 for m in generated if m.get("loss") != 0),
                "masked_generated": sum(1 for m in generated if m.get("loss") == 0),
                "lost_outputs": len(lost_outputs),
                "generated_by_model": dict(Counter(m.get("source_model") or "unknown" for m in generated if m.get("loss") != 0)),
                "preview": message_preview(messages),
            }
        )
    return sorted(samples, key=lambda s: s["trace_id"])


def selected_terminal_windows(thread_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for trace_id, trace_rows in by(thread_rows, "trace_id").items():
        llms = main_runs(trace_rows)
        if not llms:
            continue
        run = llms[-1]
        outputs = run["output"].get("messages", [])
        windows.append(
            {
                "trace_id": trace_id,
                "run_id": run["id"],
                "source_model": run["model"] or "unknown",
                "context_messages": run["input"].get("message_count", len(run["input"].get("messages", []))),
                "output_messages": len(outputs),
                "output_sigs": [sig(m) for m in outputs],
                "output_preview": message_preview(outputs),
                "start_time": run["start_time"],
            }
        )
    return sorted(windows, key=lambda w: w["start_time"])


def apply_terminal_mask(windows: list[dict[str, Any]], target_model: str) -> dict[str, Any]:
    seen: set[str] = set()
    out = {"train": 0, "model_masked": 0, "dedup_masked": 0, "outputs": 0, "by_model": Counter()}
    for window in windows:
        model = window["source_model"]
        out["by_model"][model] += window["output_messages"]
        for key in window["output_sigs"]:
            out["outputs"] += 1
            if target_model != "all" and model != target_model:
                out["model_masked"] += 1
            elif key in seen:
                out["dedup_masked"] += 1
            else:
                out["train"] += 1
                seen.add(key)
    out["by_model"] = dict(out["by_model"])
    return out


def current_target_gap(samples: list[dict[str, Any]], target_model: str) -> dict[str, int]:
    trainable = 0
    would_need_model_mask = 0
    unknown_source = 0
    for sample in samples:
        for model, count in sample["generated_by_model"].items():
            trainable += count
            if model == "unknown":
                unknown_source += count
            elif target_model != "all" and model != target_model:
                would_need_model_mask += count
    return {
        "trainable": trainable,
        "would_need_model_mask": would_need_model_mask,
        "unknown_source": unknown_source,
    }


def message_preview(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    preview = []
    for msg in messages[:limit]:
        preview.append(
            {
                "role": msg.get("role", ""),
                "len": msg.get("content_len", 0),
                "hash": msg.get("content_hash", ""),
                "blocks": "/".join(msg.get("block_types", [])),
                "markers": msg.get("markers", []),
                "loss": msg.get("loss"),
                "source_model": msg.get("source_model", ""),
            }
        )
    return preview


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    thread_cards = []
    models = sorted({r["model"] for r in rows if r.get("model")})
    targets = ["all"] + models
    for thread_id, thread_rows in by(rows, "thread_id").items():
        traces = by(thread_rows, "trace_id")
        current = trace_first_samples(thread_rows)
        terminal = selected_terminal_windows(thread_rows)
        by_target = {}
        for target in targets:
            by_target[target] = {
                "current": current_target_gap(current, target),
                "proposed": apply_terminal_mask(terminal, target),
            }
        representative_current = max(current, key=lambda s: (s["lost_outputs"], s["main_runs"]), default=None)
        representative_proposed = next((w for w in terminal if representative_current and w["trace_id"] == representative_current["trace_id"]), terminal[0] if terminal else None)
        thread_cards.append(
            {
                "thread_id": thread_id,
                "rows": len(thread_rows),
                "traces": len(traces),
                "main_llm_runs": len(main_runs(thread_rows)),
                "candidate_windows": len(main_runs(thread_rows)),
                "models": dict(Counter(r["model"] for r in thread_rows if r.get("model"))),
                "markers": dict(Counter(m for r in thread_rows for m in r.get("row_markers", []))),
                "current": {
                    "samples": len(current),
                    "messages": sum(s["messages"] for s in current),
                    "trainable_generated": sum(s["trainable_generated"] for s in current),
                    "masked_generated": sum(s["masked_generated"] for s in current),
                    "lost_outputs": sum(s["lost_outputs"] for s in current),
                    "representative": representative_current,
                },
                "proposed": {
                    "selected_windows": len(terminal),
                    "context_messages": sum(w["context_messages"] for w in terminal),
                    "output_messages": sum(w["output_messages"] for w in terminal),
                    "representative": representative_proposed,
                },
                "by_target": by_target,
            }
        )
    thread_cards.sort(key=lambda x: x["thread_id"])
    totals = {
        "rows": len(rows),
        "threads": len({r["thread_id"] for r in rows}),
        "traces": len({r["trace_id"] for r in rows}),
        "main_llm_runs": sum(c["main_llm_runs"] for c in thread_cards),
        "current_samples": sum(c["current"]["samples"] for c in thread_cards),
        "proposed_windows": sum(c["proposed"]["selected_windows"] for c in thread_cards),
        "candidate_windows": sum(c["candidate_windows"] for c in thread_cards),
    }
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target_default": TARGET_DEFAULT if TARGET_DEFAULT in models else (models[0] if models else "all"),
        "targets": targets,
        "totals": totals,
        "threads": thread_cards,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangSmith pipeline 简化对比</title>
  <style>
    :root { --bg:#f7f8fb; --panel:#fff; --ink:#182033; --muted:#667085; --line:#d7deea; --green:#087443; --red:#b42318; --blue:#245dc1; --amber:#946200; --soft:#eef2f7; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); line-height:1.45; }
    header { background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
    .bar { max-width:1280px; margin:0 auto; padding:16px 20px; display:grid; grid-template-columns:1fr auto; gap:14px; align-items:center; }
    h1 { margin:0; font-size:21px; letter-spacing:0; }
    .sub { margin-top:4px; color:var(--muted); font-size:13px; }
    main { max-width:1280px; margin:0 auto; padding:18px 20px 40px; }
    select { height:36px; border:1px solid var(--line); border-radius:6px; padding:0 10px; background:#fff; color:var(--ink); }
    .notice { background:#fff; border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:8px; padding:12px 14px; margin-bottom:14px; }
    .metrics { display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:10px; margin-bottom:16px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }
    .metric strong { display:block; font-size:24px; }
    .metric span { color:var(--muted); font-size:12px; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:16px; }
    .panel h2 { margin:0; padding:12px 14px; font-size:16px; border-bottom:1px solid var(--line); background:#fbfcfe; }
    .body { padding:14px; }
    .flow { display:grid; gap:8px; }
    .step { display:grid; grid-template-columns:34px 1fr; gap:10px; align-items:start; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .num { width:26px; height:26px; border-radius:50%; background:var(--soft); display:grid; place-items:center; font-weight:800; color:#344054; }
    .step strong { display:block; }
    .step span { display:block; color:var(--muted); font-size:13px; margin-top:2px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; }
    th { background:#f9fafb; color:#344054; }
    code { background:var(--soft); border-radius:4px; padding:1px 4px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
    .tag { display:inline-flex; align-items:center; min-height:22px; padding:2px 8px; border-radius:999px; background:var(--soft); color:#344054; margin:1px 3px 1px 0; font-size:12px; }
    .green { background:#e7f5ee; color:var(--green); }
    .red { background:#fdecec; color:var(--red); }
    .blue { background:#e8f0ff; color:var(--blue); }
    .amber { background:#fff4d7; color:var(--amber); }
    .thread { display:grid; gap:12px; }
    .thread-head { display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:12px 14px; background:#fbfcfe; border-bottom:1px solid var(--line); }
    .thread-title { font-size:17px; font-weight:800; margin-right:6px; }
    .sample-compare { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .mini { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }
    .mini h3 { margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid var(--line); background:#fbfcfe; }
    .mini .inside { padding:12px; }
    .msg { border:1px solid var(--line); border-radius:6px; padding:8px; margin-top:7px; background:#fff; }
    .msg strong { display:inline-block; width:84px; }
    @media (max-width:920px) { .bar, .grid2, .sample-compare { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>简化版：trace-first vs thread-first RichMessage</h1>
        <div class="sub">只看流程差异和最终结果。内容仍为去敏摘要，不嵌入原始 message 文本。生成时间：<span id="generatedAt"></span></div>
      </div>
      <div>
        <label>目标模型 </label>
        <select id="targetModel"></select>
      </div>
    </div>
  </header>
  <main>
    <div class="notice"><strong>核心结论：</strong>thread-first 不应把所有候选 run/window 都直接展示或训练。它先保留更完整的候选空间和 source metadata，随后通过 selection/dedup/model-mask 收敛到最终训练窗口；最终结果应与 trace 数同量级，而不是 87 个候选全部铺开。</div>
    <section class="metrics" id="metrics"></section>
    <section class="grid2">
      <div class="panel">
        <h2>现在：trace-first</h2>
        <div class="body flow">
          <div class="step"><div class="num">1</div><div><strong>按 trace_id 分组</strong><span>每个 trace 内收集 llm/tool/chain runs。</span></div></div>
          <div class="step"><div class="num">2</div><div><strong>取最后一个 main LLM run</strong><span>用 last.inputs + last.outputs 形成一个扁平 messages 样本。</span></div></div>
          <div class="step"><div class="num">3</div><div><strong>按边界打 loss=0</strong><span>compact/history 作为上下文；assistant/tool_call 默认可训练。</span></div></div>
          <div class="step"><div class="num">4</div><div><strong>emit 后 metadata 丢失</strong><span>source_run/source_model 不在最终 message 上，热切模型 mask 很难可靠补做。</span></div></div>
        </div>
      </div>
      <div class="panel">
        <h2>建议：thread-first RichMessage</h2>
        <div class="body flow">
          <div class="step"><div class="num">1</div><div><strong>按 thread_id 收集 raw runs</strong><span>run 仍是原子事实，thread 只是外层组织容器。</span></div></div>
          <div class="step"><div class="num">2</div><div><strong>保留 Rich metadata</strong><span>每条候选消息保留 source_run/source_trace/source_model。</span></div></div>
          <div class="step"><div class="num">3</div><div><strong>选择最终窗口</strong><span>从候选 run/window 中选 terminal 或 longest-prefix，候选不等于最终训练样本。</span></div></div>
          <div class="step"><div class="num">4</div><div><strong>再 dedup + model mask</strong><span>在 flatten 前写出 loss=0，最后 emit 成训练 JSONL。</span></div></div>
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>最终结果对比</h2>
      <div class="body"><table id="resultTable"></table></div>
    </section>
    <section class="panel">
      <h2>每个 thread 的极简样本对比</h2>
      <div class="body thread" id="threadCards"></div>
    </section>
  </main>
  <script id="summary" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('summary').textContent);
    const target = document.getElementById('targetModel');
    const short = (s, n = 8) => String(s || '').slice(0, n);
    const tag = (text, cls = '') => `<span class="tag ${cls}">${escapeHtml(text)}</span>`;
    const code = (text) => `<code>${escapeHtml(text)}</code>`;
    function escapeHtml(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
    function setup() {
      document.getElementById('generatedAt').textContent = DATA.generated_at;
      target.innerHTML = DATA.targets.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m === 'all' ? '不按模型 mask' : m)}</option>`).join('');
      target.value = DATA.targets.includes(DATA.target_default) ? DATA.target_default : DATA.targets[0];
      target.addEventListener('change', render);
      render();
    }
    function render() {
      const model = target.value;
      renderMetrics(model);
      renderTable(model);
      renderThreads(model);
    }
    function sumBy(fn) { return DATA.threads.reduce((n, t) => n + fn(t), 0); }
    function renderMetrics(model) {
      const proposedTrain = sumBy(t => t.by_target[model].proposed.train);
      const proposedMasked = sumBy(t => t.by_target[model].proposed.model_masked + t.by_target[model].proposed.dedup_masked);
      const currentNeedMask = sumBy(t => t.by_target[model].current.would_need_model_mask);
      const items = [
        ['raw rows', DATA.totals.rows],
        ['threads / traces', `${DATA.totals.threads} / ${DATA.totals.traces}`],
        ['candidate runs', DATA.totals.candidate_windows],
        ['final samples', `${DATA.totals.current_samples} / ${DATA.totals.proposed_windows}`],
        ['proposed train/mask', `${proposedTrain} / ${proposedMasked}`],
      ];
      document.getElementById('metrics').innerHTML = items.map(([k,v]) => `<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join('');
    }
    function renderTable(model) {
      const rows = DATA.threads.map(t => {
        const c = t.by_target[model].current;
        const p = t.by_target[model].proposed;
        return `<tr>
          <td>${code(short(t.thread_id, 8))}</td>
          <td>${t.rows}</td>
          <td>${t.traces}</td>
          <td>${t.main_llm_runs}</td>
          <td>${t.current.samples} samples<br>${tag(`${t.current.trainable_generated} trainable`, 'green')} ${tag(`${t.current.lost_outputs} lost earlier outputs`, t.current.lost_outputs ? 'red' : '')}</td>
          <td>${t.proposed.selected_windows} windows<br>${tag(`${p.train} train`, 'green')} ${tag(`${p.model_masked} model-mask`, p.model_masked ? 'red' : '')} ${tag(`${p.dedup_masked} dedup`, p.dedup_masked ? 'amber' : '')}</td>
          <td>${c.would_need_model_mask ? tag(`${c.would_need_model_mask} current tokens need source_model`, 'red') : tag('no mismatch for target', 'green')}</td>
        </tr>`;
      }).join('');
      document.getElementById('resultTable').innerHTML = `<thead><tr>
        <th>thread</th><th>rows</th><th>traces</th><th>candidate main runs</th><th>trace-first final</th><th>thread-first final</th><th>关键差异</th>
      </tr></thead><tbody>${rows}</tbody>`;
    }
    function renderThreads(model) {
      document.getElementById('threadCards').innerHTML = DATA.threads.map(t => {
        const c = t.current.representative;
        const p = t.proposed.representative;
        const pStats = t.by_target[model].proposed;
        return `<div class="panel">
          <div class="thread-head">
            <span class="thread-title">${escapeHtml(t.thread_id)}</span>
            ${tag(`${t.traces} traces`, 'blue')}
            ${tag(`${t.main_llm_runs} candidate runs`)}
            ${Object.entries(t.models).slice(0,4).map(([m,n]) => tag(`${m}=${n}`)).join('')}
          </div>
          <div class="body sample-compare">
            <div class="mini">
              <h3>trace-first 展示：1 个 trace 样本</h3>
              <div class="inside">
                ${c ? `<p>${code(short(c.trace_id,8))} ${tag(`last=${c.last_model}`, 'blue')} ${tag(`${c.messages} messages`)} ${tag(`${c.trainable_generated} trainable`, 'green')} ${tag(`${c.lost_outputs} earlier outputs absent`, c.lost_outputs ? 'red' : '')}</p>
                ${renderPreview(c.preview)}` : '<p>no sample</p>'}
              </div>
            </div>
            <div class="mini">
              <h3>thread-first 最终：selected terminal window</h3>
              <div class="inside">
                ${p ? `<p>${code(short(p.trace_id,8))} ${tag(`run=${short(p.run_id,8)}`)} ${tag(`model=${p.source_model}`, 'blue')} ${tag(`${p.context_messages} context msg`, 'red')} ${tag(`${p.output_messages} output msg`, p.source_model === model || model === 'all' ? 'green' : 'red')}</p>
                <p>${tag(`${pStats.train} train after mask`, 'green')} ${tag(`${pStats.model_masked} model masked`, pStats.model_masked ? 'red' : '')} ${tag(`${pStats.dedup_masked} dedup masked`, pStats.dedup_masked ? 'amber' : '')}</p>
                ${renderPreview(p.output_preview)}` : '<p>no selected window</p>'}
              </div>
            </div>
          </div>
        </div>`;
      }).join('');
    }
    function renderPreview(messages) {
      if (!messages || !messages.length) return '<p class="sub">no preview</p>';
      return messages.map(m => `<div class="msg"><strong>${escapeHtml(m.role)}</strong> len=${m.len}, hash=${m.hash}, blocks=${escapeHtml(m.blocks || '')}
        <div>${(m.markers || []).map(x => tag(x, 'amber')).join('')} ${m.source_model ? tag(`src=${m.source_model}`, 'blue') : ''}</div></div>`).join('');
    }
    setup();
  </script>
</body>
</html>
"""


def main() -> None:
    _, selected_rows = read_rows()
    data = build_summary(selected_rows)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(data["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
