#!/usr/bin/env python3
"""Loopback-only browser UI for a private P1 dataset review packet.

The server intentionally binds to 127.0.0.1. Use an SSH local-port forward;
do not expose the packet, which contains private person images, on a LAN/WAN.
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MODEL_CLASSES = (
    "one", "two", "three", "four", "five",
    "like", "ok", "no_gesture", "call", "dislike",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / unquote(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("unsafe path")
    return path


def _atomic_write_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".review-", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temp = Path(handle.name)
        handle.write(data)
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def _page(manifest: dict, form: dict) -> str:
    samples = json.dumps(manifest["samples"], ensure_ascii=False).replace("<", "\\u003c")
    initial = json.dumps(form, ensure_ascii=False).replace("<", "\\u003c")
    names = json.dumps(MODEL_CLASSES, ensure_ascii=False)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>HaGRID P1 QA</title>
<style>
body{{font-family:system-ui,sans-serif;margin:18px;background:#f5f6f8;color:#1f2937}}
header{{position:sticky;top:0;z-index:2;background:#fff;padding:12px;border:1px solid #d7dce3;border-radius:8px}}
input{{padding:6px;margin-right:10px}}button{{padding:6px 10px;margin:2px;border:1px solid #aab2bd;border-radius:5px;background:#fff;cursor:pointer}}
button.active{{background:#166534;color:#fff;border-color:#166534}}button.bad{{background:#991b1b;color:#fff;border-color:#991b1b}}
#save{{background:#14532d;color:#fff;font-weight:700;padding:9px 16px}}
#msg{{margin-left:10px;font-weight:600}}h2{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #d7dce3;border-radius:8px;padding:8px}}
canvas{{width:100%;height:auto;background:#111;border-radius:4px;display:block}}
.meta{{font-size:12px;color:#4b5563;margin:5px 0}}
</style>
<header>
  <strong>P1 HaGRID QA</strong>
  <span id="count"></span><br>
  <label>审核人 <input id="reviewer" size="18"></label>
  <label>日期 <input id="reviewed_on" type="date"></label>
  <button id="bulk">我已逐张检查，全部 OK</button>
  <button id="save">保存 review.json</button><span id="msg"></span>
  <p>如果你确实已经逐张检查过全部样本，可用“全部 OK”一次性记录明确 attestation；否则请逐张选择问题类型。保存不会自动批准，后台验证通过后才会进入训练门禁。</p>
</header>
<main id="app"></main>
<script>
const samples={samples};
const initial={initial};
const names={names};
const choices=[['ok','OK'],['bad_gesture','手势错'],['bad_box','框错'],['bad_image','图片错']];
const decisions={{}};
let bulkAttestation=initial.bulk_attestation||null;
for (const row of (initial.decisions||[])) decisions[row.packet_id]=row.decision||'';
document.querySelector('#reviewer').value=initial.reviewer||'';
document.querySelector('#reviewed_on').value=initial.reviewed_on||'';
document.querySelector('#count').textContent=` — ${{samples.length}} 张`;
function esc(s){{return String(s).replace(/[&<>"']/g,x=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[x]));}}
function draw(sample, canvas){{
  const img=new Image();
  img.onload=()=>{{
    const scale=Math.min(1,720/img.naturalWidth); canvas.width=img.naturalWidth*scale; canvas.height=img.naturalHeight*scale;
    const ctx=canvas.getContext('2d'); ctx.drawImage(img,0,0,canvas.width,canvas.height);
    ctx.lineWidth=3; ctx.font='bold 14px sans-serif';
    for(const line of (sample.label_lines||[])){{
      const p=line.trim().split(/\\s+/).map(Number); if(p.length<5||p.some(Number.isNaN)) continue;
      const [cls,cx,cy,w,h]=p; const x=(cx-w/2)*canvas.width, y=(cy-h/2)*canvas.height;
      ctx.strokeStyle='#00ff65'; ctx.strokeRect(x,y,w*canvas.width,h*canvas.height);
      ctx.fillStyle='#00ff65'; ctx.fillText(names[cls]||String(cls),x+3,Math.max(15,y+15));
    }};
  }};
  img.src='/images/'+encodeURIComponent(sample.packet_id);
}}
function render(){{
  const app=document.querySelector('#app'); let current='';
  for(const s of samples){{
    if(s.class!==current){{ current=s.class; app.insertAdjacentHTML('beforeend',`<h2>${{esc(current)}}</h2><section class="grid" data-class="${{esc(current)}}"></section>`); }}
    const grid=app.querySelector(`section[data-class="${{CSS.escape(current)}}"]`);
    const card=document.createElement('article'); card.className='card';
    const canvas=document.createElement('canvas');
    const meta=document.createElement('div'); meta.className='meta'; meta.textContent=`${{s.packet_id}} · ${{s.split}}`;
    const controls=document.createElement('div');
    for(const [value,label] of choices){{
      const b=document.createElement('button'); b.textContent=label; b.dataset.value=value;
      b.onclick=()=>{{ decisions[s.packet_id]=value; controls.querySelectorAll('button').forEach(x=>x.classList.remove('active','bad')); b.classList.add(value==='ok'?'active':'bad'); }};
      if(decisions[s.packet_id]===value) b.classList.add(value==='ok'?'active':'bad');
      controls.appendChild(b);
    }}
    card.append(canvas,meta,controls); grid.appendChild(card); draw(s,canvas);
  }}
}}
render();
document.querySelector('#bulk').onclick=()=>{{
  if(!confirm('我已实际逐张查看全部 240 张样本，且全部手势、框和图片均确认 OK。继续吗？')) return;
  samples.forEach(s=>decisions[s.packet_id]='ok');
  bulkAttestation={{confirmed:true,statement:'I reviewed every sampled image and all are OK'}};
  document.querySelector('#msg').textContent='已标记全部 OK，请点击保存';
}};
document.querySelector('#save').onclick=async()=>{{
  const payload={{...initial, reviewer:document.querySelector('#reviewer').value, reviewed_on:document.querySelector('#reviewed_on').value,
    bulk_attestation:bulkAttestation,
    decisions:samples.map(s=>({{packet_id:s.packet_id,class:s.class,decision:decisions[s.packet_id]||'',notes:''}}))}};
  const msg=document.querySelector('#msg'); msg.textContent='保存中…';
  try{{const r=await fetch('/api/review',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}}); const d=await r.json(); if(!r.ok) throw new Error(d.error||'保存失败'); msg.textContent='已保存；后台会自动验证';}}
  catch(e){{msg.textContent='错误：'+e.message;}}
}};
</script>"""


class ReviewHandler(BaseHTTPRequestHandler):
    workspace: Path

    def log_message(self, _format: str, *_args) -> None:
        # Do not log reviewer identity or request payloads.
        return

    def _json(self, value: dict, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                manifest = _read_json(self.workspace / "packet.manifest.json")
                form = _read_json(self.workspace / "review.json")
                data = _page(manifest, form).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif path.startswith("/images/"):
                file = _safe_child(self.workspace / "images", path[len("/images/"):])
                data = file.read_bytes()
                content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            elif path == "/api/state":
                self._json(_read_json(self.workspace / "review.json"))
                return
            else:
                self._json({"error": "not found"}, 404)
                return
        except (OSError, ValueError, json.JSONDecodeError):
            self._json({"error": "resource unavailable"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/review":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid body size")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            current = _read_json(self.workspace / "review.json")
            if value.get("packet_manifest_sha256") != current.get("packet_manifest_sha256"):
                raise ValueError("packet manifest binding changed")
            decisions = value.get("decisions")
            if not isinstance(decisions, list) or len(decisions) != 240:
                raise ValueError("exactly 240 decisions are required")
            _atomic_write_json(self.workspace / "review.json", value)
            self._json({"status": "SAVED"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    required = (workspace / "packet.manifest.json", workspace / "review.json", workspace / "images")
    if not all(path.exists() for path in required):
        raise SystemExit("QA workspace is incomplete")
    handler = type("BoundReviewHandler", (ReviewHandler,), {"workspace": workspace})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"QA_SERVER_READY http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
