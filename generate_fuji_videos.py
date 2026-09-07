#!/usr/bin/env python3
"""Generate four Mount Fuji history clips through Hugging Face Inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "ali-vilab/text-to-video-ms-1.7b"
DEFAULT_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}"
MP4_HEADER_BYTES = 32

SCENES = {
    "scene_01_birth_of_fuji.mp4": (
        "Hyper-realistic 3D geological timelapse of Mount Fuji's creation. "
        "Millions of years of volcanic layers piling up dynamically. Ancient "
        "Japanese landscape with surrounding Suruga Bay and primitive land "
        "shifting rapidly. The symmetric cone of Mount Fuji emerging from "
        "tectonic movements, extreme detail, photorealistic, National "
        "Geographic documentary style, 8k resolution, cinematic lighting, 24fps."
    ),
    "scene_02_eruptions_and_faith.mp4": (
        "Historical dynamic wide shot of Mount Fuji during the Heian and "
        "Muromachi periods. Multiple flank eruptions bursting from the "
        "mountainside with glowing red magma and thick smoke. In the lush "
        "forest below, ancient Shinto aesthetics, a small wooden Asama Shrine "
        "surrounded by mountain ascetics (shugenja) praying. Mythical, dark "
        "atmosphere, volumetric fog, realistic fire and smoke simulation, "
        "cinematic, 8k."
    ),
    "scene_03_1707_hoei_eruption.mp4": (
        "Epic cinematic recreation of the 1707 Hoei Eruption of Mount Fuji. A "
        "massive blast on the right flank of the mountain, creating the giant "
        "Hoei crater. Plumes of black volcanic ash erupting into the "
        "stratosphere, blocking the sun. Dark apocalyptic sky, lightning "
        "flashing in the ash cloud, realistic dust and particle physics, "
        "dramatic, historical disaster movie style, ultra-detailed, 8k."
    ),
    "scene_04_modern_seasons.mp4": (
        "Breathtaking high-altitude time-lapse of modern Mount Fuji and Fuji "
        "Five Lakes. Fast-moving volumetric clouds, sun rising behind the "
        "mountain summit creating a beautiful Goraiko glow. Transitioning "
        "through seasons: snow capped winter, green summer with trees, and "
        "autumn colors reflected on the lake surface. Drone view, pristine "
        "nature, peaceful, majestic, hyper-realistic, 8k."
    ),
}


class GenerationError(RuntimeError):
    """The remote inference service did not return an MP4."""


def _error_message(body: bytes) -> str:
    try:
        payload = json.loads(body)
        return str(payload.get("error", payload))
    except (ValueError, AttributeError):
        return body.decode("utf-8", errors="replace")[:500]


def _is_mp4(body: bytes) -> bool:
    """Return whether *body* starts with an ISO Base Media File Type box."""
    return len(body) >= 12 and body[4:8] == b"ftyp"


def generate(prompt: str, token: str, model: str, timeout: int, retries: int) -> bytes:
    request = urllib.request.Request(
        DEFAULT_ENDPOINT.format(model=model),
        data=json.dumps({"inputs": prompt}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "video/mp4",
            "User-Agent": "fuji-history-video-generator/1.0",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                video = response.read()
                content_type = response.headers.get_content_type()
            # Content-Type is advisory: error proxies occasionally preserve the
            # requested video/mp4 type.  The file signature is the authoritative
            # check and also allows application/octet-stream responses.
            if not _is_mp4(video[:MP4_HEADER_BYTES]):
                raise GenerationError(
                    f"unexpected response ({content_type}): {_error_message(video)}"
                )
            return video
        except urllib.error.HTTPError as exc:
            message = _error_message(exc.read())
            if exc.code not in (429, 503) or attempt == retries:
                raise GenerationError(f"HTTP {exc.code}: {message}") from exc
            time.sleep(min(60, 5 * (2**attempt)))
        except urllib.error.URLError as exc:
            if attempt == retries:
                raise GenerationError(f"network error: {exc.reason}") from exc
            time.sleep(min(60, 5 * (2**attempt)))
    raise AssertionError("retry loop exited unexpectedly")


def save_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--retries", type=nonnegative_int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="print work without API calls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not args.dry_run and not token:
        print("error: set HF_TOKEN to a Hugging Face access token", file=sys.stderr)
        return 2
    for index, (filename, prompt) in enumerate(SCENES.items(), 1):
        destination = args.output_dir / filename
        if args.dry_run:
            print(f"[{index}/4] {destination}: {prompt}")
            continue
        print(f"[{index}/4] Generating {destination} ...", flush=True)
        try:
            video = generate(prompt, token, args.model, args.timeout, args.retries)
            save_atomic(destination, video)
        except GenerationError as exc:
            print(f"error: {filename}: {exc}", file=sys.stderr)
            return 1
        print(f"[{index}/4] Saved {len(video):,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
