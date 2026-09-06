"""Create a standalone per-frame viewer for a recorded RoboCasa rollout."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .robocasa_rollout import RoboCasaEpisode, _slug


def _image_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("could not encode rollout image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def build_rollout_viewer(
    episode_dir: str | Path,
    memory_dir: str | Path,
    *,
    max_points: int = 1800,
) -> str:
    episode = RoboCasaEpisode(episode_dir)
    memory_root = Path(memory_dir)
    frames = []
    for index in range(len(episode)):
        frame = episode.frame(index)
        arrays = frame["arrays"]
        images = [
            {"name": camera.removeprefix("video."), "url": _image_url(arrays[f"rgb__{_slug(camera)}"])}
            for camera in episode.manifest["cameras"]
        ]
        memory = json.loads((memory_root / f"memory_frame_{index:06d}.json").read_text(encoding="utf-8"))
        tokens = memory["tokens"]
        if len(tokens) > max_points:
            step = len(tokens) / max_points
            tokens = [tokens[int(i * step)] for i in range(max_points)]
        actions = {
            key.removeprefix("action__").replace("__", "."): arrays[key].reshape(-1).round(4).tolist()
            for key in arrays if key.startswith("action__")
        }
        frames.append(
            {
                "index": index,
                "reward": frame["reward"],
                "success": frame["success"],
                "source": frame.get("action_source", {}),
                "images": images,
                "actions": actions,
                "tokens": [
                    {
                        "p": token["position"],
                        "c": token["feature"][:3],
                        "q": token["confidence"],
                        "current": token["last_seen"] == index,
                    }
                    for token in tokens
                ],
            }
        )
    payload = json.dumps({"task": episode.manifest["task"], "frames": frames}, separators=(",", ":"))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboCasa rollout viewer</title><style>
:root{{--bg:#08101d;--panel:#111c2c;--line:#263850;--text:#eef6ff;--muted:#8ea4bc;--green:#3ce3a4;--orange:#ffb35c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}h1{{margin:0 0 5px}}#task{{color:var(--muted)}}.controls{{display:flex;align-items:center;gap:12px;margin:18px 0}}button,input{{accent-color:#56a8ff}}button{{background:#1b3049;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 11px}}input{{flex:1}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}}.camera img{{width:100%;display:block;border-radius:7px}}.camera b{{display:block;margin-bottom:8px}}.lower{{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin-top:10px}}canvas{{width:100%;height:390px;background:#050b13;border-radius:7px}}pre{{white-space:pre-wrap;color:#c7d6e8;margin:0;max-height:390px;overflow:auto}}.legend{{color:var(--muted);margin-top:7px}}@media(max-width:850px){{.grid,.lower{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>RoboCasa rollout</h1><div id="task"></div><div class="controls"><button id="prev">Previous</button><input id="frame" type="range" min="0"><button id="next">Next</button><b id="counter"></b></div><section class="grid" id="cameras"></section><section class="lower"><div class="card"><canvas id="cloud"></canvas><div class="legend">XY world projection · green=current · orange=persisted · brightness=confidence</div></div><div class="card"><pre id="details"></pre></div></section></main><script>
const D={payload}, slider=document.querySelector('#frame');slider.max=D.frames.length-1;document.querySelector('#task').textContent=D.task;
function bounds(points,a,b){{const xs=points.map(x=>x.p[a]),ys=points.map(x=>x.p[b]);return [Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)]}}
function cloud(tokens){{const c=document.querySelector('#cloud'),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,r.width,r.height);if(!tokens.length)return;const B=bounds(tokens,0,1),pad=18,sx=(r.width-2*pad)/Math.max(B[1]-B[0],1e-6),sy=(r.height-2*pad)/Math.max(B[3]-B[2],1e-6),s=Math.min(sx,sy);for(const t of tokens){{const q=.25+.75*t.q;x.fillStyle=t.current?`rgba(60,227,164,${{q}})`:`rgba(255,179,92,${{q}})`;x.fillRect(pad+(t.p[0]-B[0])*s,r.height-pad-(t.p[1]-B[2])*s,2.2,2.2)}}}}
function render(){{const f=D.frames[+slider.value];document.querySelector('#counter').textContent=`Frame ${{f.index+1}} / ${{D.frames.length}}`;document.querySelector('#cameras').innerHTML=f.images.map(i=>`<div class="card camera"><b>${{i.name}}</b><img src="${{i.url}}"></div>`).join('');document.querySelector('#details').textContent=JSON.stringify({{reward:f.reward,success:f.success,action_source:f.source,actions:f.actions,memory_points:f.tokens.length}},null,2);cloud(f.tokens)}}
document.querySelector('#prev').onclick=()=>{{slider.value=Math.max(0,+slider.value-1);render()}};document.querySelector('#next').onclick=()=>{{slider.value=Math.min(+slider.max,+slider.value+1);render()}};slider.oninput=render;window.onresize=render;render();
</script></body></html>"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_rollout_viewer(args.episode_dir, args.memory_dir), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
