"""Create a position-randomized, browser-local blind A/B annotation page."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rel(path: str | Path, page_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), page_dir.resolve())).as_posix()


def build(pairs: list[dict[str, Any]], out_dir: Path, videos_dir: Path, seed: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    public: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for pair in pairs:
        good = videos_dir / pair["good_record_id"] / "generate.mp4"
        bad = videos_dir / pair["bad_record_id"] / "generate.mp4"
        if not good.is_file() or not bad.is_file():
            raise FileNotFoundError(f"missing rendered pair {pair['pair_id']}: {good} / {bad}")
        good_side = rng.choice(("A", "B"))
        video_a, video_b = (good, bad) if good_side == "A" else (bad, good)
        public_a = media_dir / f"{pair['pair_id']}_A.mp4"
        public_b = media_dir / f"{pair['pair_id']}_B.mp4"
        shutil.copy2(video_a, public_a)
        shutil.copy2(video_b, public_b)
        public.append(
            {
                "pair_id": pair["pair_id"],
                "instruction": pair["instruction"],
                "source_video": _rel(pair["source_video"], out_dir),
                "video_a": _rel(public_a, out_dir),
                "video_b": _rel(public_b, out_dir),
            }
        )
        key.append({**pair, "good_side": good_side, "video_a": str(video_a), "video_b": str(video_b)})

    data = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aurora Week 1 Blind A/B</title>
<style>
:root{{--bg:#0c111b;--card:#151d2b;--muted:#94a3b8;--text:#edf2f7;--accent:#60a5fa}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#0c111beF;padding:16px 24px;border-bottom:1px solid #273244}}
.toolbar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}button{{background:var(--accent);border:0;border-radius:8px;padding:9px 14px;font-weight:700;cursor:pointer}}#progress{{color:var(--muted)}}main{{max-width:1500px;margin:auto;padding:20px}}
.card{{background:var(--card);border:1px solid #273244;border-radius:14px;margin:0 0 20px;padding:18px}}h2{{margin:0 0 5px;font-size:18px}}.instruction{{font-size:17px;margin:8px 0 16px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{margin:0}}figcaption{{color:var(--muted);font-weight:700;margin-bottom:6px}}video{{width:100%;background:#000;border-radius:8px;aspect-ratio:16/10}}.vote{{display:flex;gap:18px;margin-top:15px;align-items:center;flex-wrap:wrap}}label{{cursor:pointer}}textarea{{width:100%;min-height:60px;margin-top:12px;background:#0d1420;color:var(--text);border:1px solid #354258;border-radius:7px;padding:9px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="toolbar"><strong>Aurora · 盲评</strong><span id="progress"></span><button id="export">导出 JSONL</button><button id="clear">清空本地标注</button></div></header><main id="app"></main>
<script>const pairs={data};const storageKey='aurora-week1-ab-{seed}';let saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function persist(){{localStorage.setItem(storageKey,JSON.stringify(saved));const done=Object.values(saved).filter(x=>x.human_label).length;document.querySelector('#progress').textContent=`${{done}} / ${{pairs.length}} 已标注`;}}
function render(){{document.querySelector('#app').innerHTML=pairs.map((p,i)=>{{const s=saved[p.pair_id]||{{}};return `<section class="card"><h2>${{i+1}} / ${{pairs.length}} · ${{esc(p.pair_id)}}</h2><div class="instruction">指令：${{esc(p.instruction)}}</div><div class="grid"><figure><figcaption>原视频</figcaption><video controls loop muted preload="metadata" src="${{esc(p.source_video)}}"></video></figure><figure><figcaption>候选 A</figcaption><video controls loop muted preload="metadata" src="${{esc(p.video_a)}}"></video></figure><figure><figcaption>候选 B</figcaption><video controls loop muted preload="metadata" src="${{esc(p.video_b)}}"></video></figure></div><div class="vote">${{['A','B','tie','invalid'].map(x=>`<label><input type="radio" name="${{p.pair_id}}" value="${{x}}" ${{s.human_label===x?'checked':''}}> ${{x}}</label>`).join('')}}</div><textarea data-note="${{p.pair_id}}" placeholder="可选：判断依据或异常说明">${{esc(s.human_notes||'')}}</textarea></section>`}}).join('');
document.querySelectorAll('input[type=radio]').forEach(el=>el.onchange=e=>{{const id=e.target.name;saved[id]={{...(saved[id]||{{}}),human_label:e.target.value}};persist()}});document.querySelectorAll('textarea').forEach(el=>el.oninput=e=>{{const id=e.target.dataset.note;saved[id]={{...(saved[id]||{{}}),human_notes:e.target.value}};persist()}});persist();}}
document.querySelector('#export').onclick=()=>{{const lines=pairs.map(p=>JSON.stringify({{bench_id:p.pair_id,human_label:(saved[p.pair_id]||{{}}).human_label||'',human_notes:(saved[p.pair_id]||{{}}).human_notes||''}})).join('\\n')+'\\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{{type:'application/jsonl'}}));a.download='judge_agreement_human.jsonl';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#clear').onclick=()=>{{if(confirm('确认清空这 40 组的本地标注？')){{saved={{}};persist();render()}}}};render();</script></body></html>"""
    page_path = out_dir / "blind_eval.html"
    key_path = out_dir / "blind_key.json"
    page_path.write_text(page, encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return page_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    page, key = build(load_jsonl(args.pairs), args.out_dir, args.videos_dir, args.seed)
    print(json.dumps({"page": str(page), "key": str(key)}, indent=2))


if __name__ == "__main__":
    main()
