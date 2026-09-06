"""Build a standalone interactive dashboard for the current policy gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dashboard(base: dict, adapted: dict, gate: dict) -> str:
    labels = {
        "base": "Base SmolVLA",
        "expert": "Expert adapted",
        "residual": "Residual policy",
        "hold": "Hold-state",
    }
    colors = {
        "base": "#718096",
        "expert": "#4f8cff",
        "residual": "#9b6cff",
        "hold": "#2dd4a7",
    }
    metric_names = {
        "mae": "MAE",
        "rmse": "RMSE",
        "normalized_rmse": "Normalized RMSE",
    }
    aggregate = {
        "base": base["aggregate"]["smolvla"],
        "expert": adapted["aggregate"]["smolvla"],
        "residual": adapted["aggregate"]["residual_blend"],
        "hold": adapted["aggregate"]["hold_state_baseline"],
    }
    episodes = {}
    for episode in adapted["episodes"]:
        key = str(episode)
        episodes[key] = {
            "base": base["per_episode"][key]["smolvla"],
            "expert": adapted["per_episode"][key]["smolvla"],
            "residual": adapted["per_episode"][key]["residual_blend"],
            "hold": adapted["per_episode"][key]["hold_state_baseline"],
        }
    payload = json.dumps(
        {
            "labels": labels,
            "colors": colors,
            "metricNames": metric_names,
            "aggregate": aggregate,
            "episodes": episodes,
            "gate": gate,
        },
        separators=(",", ":"),
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmolVLA policy gate</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#131a2d; --muted:#9aa8c2; --text:#edf2ff; --line:#27314b; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font:15px/1.45 system-ui,sans-serif; background:radial-gradient(circle at top,#18213b 0,var(--bg) 45%); color:var(--text) }}
main {{ max-width:1180px; margin:auto; padding:32px 22px 48px }}
.top {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; flex-wrap:wrap }}
h1 {{ margin:0 0 5px; font-size:clamp(25px,4vw,40px) }}
.subtitle,.note {{ color:var(--muted) }}
.badge {{ padding:8px 13px; border:1px solid #ff758f; color:#ff9aad; background:#3b1723; border-radius:999px; font-weight:700 }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin:25px 0 }}
.card,.panel {{ background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line); border-radius:14px; box-shadow:0 16px 40px #0004 }}
.card {{ padding:17px }} .card span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em }}
.card strong {{ display:block; margin-top:6px; font-size:25px }}
.panel {{ padding:20px; margin-top:14px }}
.panel-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap }}
h2 {{ margin:0; font-size:18px }}
select {{ color:var(--text); background:#1c2741; border:1px solid #405071; border-radius:8px; padding:8px 11px }}
.legend {{ display:flex; flex-wrap:wrap; gap:15px; margin:16px 0 3px; color:var(--muted); font-size:13px }}
.dot {{ width:10px; height:10px; border-radius:3px; display:inline-block; margin-right:6px }}
.chart {{ min-height:270px; display:flex; align-items:flex-end; gap:18px; padding:28px 10px 10px; border-bottom:1px solid var(--line) }}
.group {{ flex:1; min-width:80px; height:235px; display:flex; align-items:flex-end; justify-content:center; gap:5px; position:relative; padding-bottom:34px }}
.bar {{ width:min(42px,22%); min-height:2px; border-radius:7px 7px 2px 2px; position:relative; transition:height .35s ease,filter .2s }}
.bar:hover {{ filter:brightness(1.25) }}
.bar b {{ position:absolute; width:70px; left:50%; transform:translateX(-50%); top:-23px; font-size:11px; text-align:center }}
.group-label {{ position:absolute; bottom:4px; left:0; right:0; color:var(--muted); text-align:center; font-size:12px }}
.explain {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:14px }}
.callout {{ padding:15px; border-left:3px solid #9b6cff; background:#171d33; border-radius:7px }}
@media(max-width:720px) {{ .cards {{ grid-template-columns:1fr 1fr }} .explain {{ grid-template-columns:1fr }} .chart {{ overflow-x:auto }} .group {{ min-width:105px }} }}
</style>
</head>
<body><main>
<div class="top"><div><h1>SmolVLA policy gate</h1><div class="subtitle">Frozen held-out evaluation · 6 episodes · 900 predicted actions</div></div><div class="badge">Gate not passed</div></div>
<section class="cards">
 <div class="card"><span>Best policy RMSE</span><strong>12.1506</strong></div>
 <div class="card"><span>Hold-state RMSE</span><strong>11.3306</strong></div>
 <div class="card"><span>Gap to control</span><strong>+7.2%</strong></div>
 <div class="card"><span>Episode RMSE wins</span><strong>2 / 6</strong></div>
</section>
<section class="panel"><div class="panel-head"><h2>Aggregate comparison</h2><select id="metric"><option value="rmse">RMSE</option><option value="normalized_rmse">Normalized RMSE</option><option value="mae">MAE</option></select></div><div class="legend" id="legend"></div><div class="chart" id="aggregate"></div></section>
<section class="panel"><div class="panel-head"><h2>Episode-by-episode</h2><span class="note">Lower is better</span></div><div class="chart" id="episodes"></div></section>
<div class="explain"><div class="callout"><strong>What improved</strong><br><span class="note">The residual policy cuts RMSE 46.0% relative to base SmolVLA.</span></div><div class="callout"><strong>Why fusion is still gated</strong><br><span class="note">It remains worse than hold-state in aggregate and wins only two episodes on RMSE.</span></div></div>
</main>
<script>
const D={payload}; const order=['base','expert','residual','hold'];
const legend=document.querySelector('#legend');
legend.innerHTML=order.map(k=>`<span><i class="dot" style="background:${{D.colors[k]}}"></i>${{D.labels[k]}}</span>`).join('');
function bars(container, groups, metric) {{
 const values=groups.flatMap(g=>order.map(k=>g.values[k][metric])); const max=Math.max(...values)*1.12;
 container.innerHTML=groups.map(g=>`<div class="group">${{order.map(k=>{{const v=g.values[k][metric];return `<div class="bar" title="${{D.labels[k]}}: ${{v.toFixed(4)}}" style="height:${{Math.max(2,v/max*190)}}px;background:${{D.colors[k]}}"><b>${{v.toFixed(metric==='normalized_rmse'?3:2)}}</b></div>`}}).join('')}}<div class="group-label">${{g.label}}</div></div>`).join('');
}}
function render() {{ const m=document.querySelector('#metric').value; bars(document.querySelector('#aggregate'),[{{label:D.metricNames[m],values:D.aggregate}}],m); bars(document.querySelector('#episodes'),Object.entries(D.episodes).map(([k,v])=>({{label:'Episode '+k,values:v}})),m); }}
document.querySelector('#metric').addEventListener('change',render); render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    html = build_dashboard(
        json.loads(args.base.read_text(encoding="utf-8")),
        json.loads(args.adapted.read_text(encoding="utf-8")),
        json.loads(args.gate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
