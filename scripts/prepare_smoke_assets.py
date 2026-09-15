#!/usr/bin/env python3
"""Create ten tiny, licensed smoke-test videos for the Aurora pipeline."""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "smoke"
IMAGE_DIR = DATA_DIR / "source_images"
VIDEO_DIR = DATA_DIR / "videos"
MANIFEST_PATH = DATA_DIR / "assets_manifest.json"
USER_AGENT = "Aurora-OutcomeDPO/0.1 (research smoke-test asset preparation)"

ASSETS = [
    ("smoke_001_dog", "golden retriever dog grass", None),
    ("smoke_002_bottle", "person holding water bottle", "File:Bottled water.jpg"),
    ("smoke_003_car", "red car street", "File:25th Street. San Francisco, CA - Red Car, Chase Bank.jpg"),
    ("smoke_004_cup", "coffee mug wooden table", "File:Mug of Coffee on a Wooden Table (26804116318).jpg"),
    ("smoke_005_city", "city skyline daylight", None),
    ("smoke_006_person", "person walking outdoors", None),
    ("smoke_007_beach", "sandy beach ocean", None),
    ("smoke_008_bicycle", "bicycle street", None),
    ("smoke_009_cat", "domestic cat sitting couch", "File:Cat-on-couch.jpg"),
    ("smoke_010_tree", "autumn tree park", None),
]

ALLOWED_LICENSE_MARKERS = ("public domain", "cc0", "cc by", "cc-by")


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def clean_html(value: str) -> str:
    import html
    import re

    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def find_commons_image(query: str, preferred_title: str | None = None) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "768",
    }
    if preferred_title:
        params["titles"] = preferred_title
    else:
        params.update(
            {
                "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": "6",
                "gsrlimit": "12",
            }
        )
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    payload = request_json(url)
    pages = sorted(payload.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 999))
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata") or {}
        license_name = clean_html((metadata.get("LicenseShortName") or {}).get("value", ""))
        usage_terms = clean_html((metadata.get("UsageTerms") or {}).get("value", ""))
        license_text = f"{license_name} {usage_terms}".lower()
        if info.get("mime") not in {"image/jpeg", "image/png"}:
            continue
        if not any(marker in license_text for marker in ALLOWED_LICENSE_MARKERS):
            continue
        return {
            "title": page["title"],
            "description_url": info.get("descriptionurl", ""),
            "download_url": info.get("thumburl") or info["url"],
            "license": license_name or usage_terms,
            "artist": clean_html((metadata.get("Artist") or {}).get("value", "")),
            "credit": clean_html((metadata.get("Credit") or {}).get("value", "")),
        }
    raise RuntimeError(f"No compatible Commons image found for query: {query}")


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg is unavailable. Run with: uv run --with imageio-ffmpeg "
            "python scripts/prepare_smoke_assets.py"
        ) from exc


def make_video(ffmpeg: str, image_path: Path, video_path: Path) -> None:
    command = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        "3",
        "-vf",
        "scale=512:320:force_original_aspect_ratio=increase,crop=512:320,format=yuv420p",
        "-r",
        "16",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[], help="Prepare only this asset id; repeat as needed")
    args = parser.parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_executable()
    existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else []
    manifest_by_id = {item["asset_id"]: item for item in existing}
    selected = [asset for asset in ASSETS if not args.only or asset[0] in set(args.only)]
    if args.only and len(selected) != len(set(args.only)):
        known = {asset[0] for asset in ASSETS}
        raise SystemExit(f"Unknown asset id(s): {sorted(set(args.only) - known)}")

    for asset_id, query, preferred_title in selected:
        print(f"Preparing {asset_id}: {query}", flush=True)
        source = find_commons_image(query, preferred_title)
        suffix = ".png" if source["download_url"].lower().split("?")[0].endswith(".png") else ".jpg"
        image_path = IMAGE_DIR / f"{asset_id}{suffix}"
        video_path = VIDEO_DIR / f"{asset_id}.mp4"
        download(source["download_url"], image_path)
        make_video(ffmpeg, image_path, video_path)
        manifest_by_id[asset_id] = {
            "asset_id": asset_id,
            "query": query,
            "video_path": str(video_path.relative_to(ROOT)),
            "source": source,
        }

    manifest = [manifest_by_id[asset_id] for asset_id, _, _ in ASSETS if asset_id in manifest_by_id]
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} video(s); manifest contains {len(manifest)} assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
