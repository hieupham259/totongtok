#!/usr/bin/env python3
"""
extend_video.py

Converted from extend_video.ipynb to a standalone Python script.

Pipeline:
1) Use a text model (Responses API) to plan N scene prompts from a base idea.
2) Render each segment with Sora; for continuity, pass the prior segment’s final frame as input_reference.
3) Concatenate segments into a single MP4.

Requirements:
- Python 3.10+
- OPENAI_API_KEY in environment (recommended)

Optional:
- Run with --install to pip-install dependencies into the current interpreter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests


# ----------------------------
# 1) Install helpers (optional)
# ----------------------------

def pip_install(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *pkgs])


def ensure(spec_name: str, *pip_pkgs: str) -> bool:
    if importlib.util.find_spec(spec_name) is None and pip_pkgs:
        pip_install(*pip_pkgs)
    return importlib.util.find_spec(spec_name) is not None


def ensure_deps(install: bool) -> None:
    """
    If install=True, attempt to install missing dependencies.
    Otherwise, validate imports and provide actionable errors.
    """
    missing: list[str] = []

    def need(mod: str, *pkgs: str) -> None:
        ok = ensure(mod, *pkgs) if install else (importlib.util.find_spec(mod) is not None)
        if not ok:
            missing.append(mod)

    # Required for pipeline
    need("openai", "openai>=1.0.0")
    need("cv2", "opencv-python-headless")
    need("imageio_ffmpeg", "imageio-ffmpeg")
    need("imageio", "imageio[ffmpeg]")

    if missing and not install:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nRe-run with --install, or install them via pip."
        )


# ----------------------------
# 2) Planner system prompt
# ----------------------------

PLANNER_SYSTEM_INSTRUCTIONS = r"""
You are a senior prompt director for Sora 2. Your job is to transform:
- a Base prompt (broad idea),
- a fixed generation length per segment (seconds),
- and a total number of generations (N),

into **N crystal-clear shot prompts** with **maximum continuity** across segments.

Rules:
1) Return **valid JSON** only. Structure:
   {
     "segments": [
       {
         "title": "Generation 1",
         "seconds": 6,
         "prompt": "<prompt block to send into Sora>"
       },
       ...
     ]
   }
   - `seconds` MUST equal the given generation length for ALL segments.
   - `prompt` should include a **Context** section for model guidance AND a **Prompt** line for the shot itself,
     exactly like in the example below.
2) Continuity:
   - Segment 1 starts fresh from the BASE PROMPT.
   - Segment k (k>1) must **begin exactly at the final frame** of segment k-1.
   - Maintain consistent visual style, tone, lighting, and subject identity unless explicitly told to change.
3) Safety & platform constraints:
   - Do not depict real people (including public figures) or copyrighted characters.
   - Avoid copyrighted music and avoid exact trademark/logos if policy disallows them; use brand-safe wording.
   - Keep content suitable for general audiences.
4) Output only JSON (no Markdown, no backticks).
5) Keep the **Context** lines inside the prompt text (they're for the AI, not visible).
6) Make the writing specific and cinematic; describe camera, lighting, motion, and subject focus succinctly.

Below is an **EXAMPLE (verbatim)** of exactly how to structure prompts with context and continuity:

Example:
Base prompt: "Intro video for the iPhone 19"
Generation length: 6 seconds each
Total generations: 3

Clearly defined prompts with maximum continuity and context:

### Generation 1:

<prompt>
First shot introducing the new iPhone 19. Initially, the screen is completely dark. The phone, positioned vertically and facing directly forward, emerges slowly and dramatically out of darkness, gradually illuminated from the center of the screen outward, showcasing a vibrant, colorful, dynamic wallpaper on its edge-to-edge glass display. The style is futuristic, sleek, and premium, appropriate for an official Apple product reveal.
<prompt>

---

### Generation 2:

<prompt>
Context (not visible in video, only for AI guidance):

* You are creating the second part of an official intro video for Apple's new iPhone 19.
* The previous 6-second scene ended with the phone facing directly forward, clearly displaying its vibrant front screen and colorful wallpaper.

Prompt: Second shot begins exactly from the final frame of the previous scene, showing the front of the iPhone 19 with its vibrant, colorful display clearly visible. Now, smoothly rotate the phone horizontally, turning it from the front to reveal the back side. Focus specifically on the advanced triple-lens camera module, clearly highlighting its premium materials, reflective metallic surfaces, and detailed lenses. Maintain consistent dramatic lighting, sleek visual style, and luxurious feel matching the official Apple product introduction theme.
</prompt>

---

### Generation 3:

<prompt>
Context (not visible in video, only for AI guidance):

* You are creating the third and final part of an official intro video for Apple's new iPhone 19.
* The previous 6-second scene ended clearly showing the back of the iPhone 19, focusing specifically on its advanced triple-lens camera module.

Prompt: Final shot begins exactly from the final frame of the previous scene, clearly displaying the back side of the iPhone 19, with special emphasis on the triple-lens camera module. Now, have a user's hand gently pick up the phone, naturally rotating it from the back to the front and bringing it upward toward their face. Clearly show the phone smoothly and quickly unlocking via Face ID recognition, transitioning immediately to a vibrant home screen filled with updated app icons. Finish the scene by subtly fading the home screen into the iconic Apple logo. Keep the visual style consistent, premium, and elegant, suitable for an official Apple product launch.
</prompt>

--

Notice how we broke up the initial prompt into multiple prompts that provide context and continuity so this all works seamlessly.
""".strip()


# ----------------------------
# 3) Low-level Sora API helpers
# ----------------------------

API_BASE = "https://api.openai.com/v1"


def guess_mime(path: Path) -> str:
    t = mimetypes.guess_type(str(path))[0]
    return t or "application/octet-stream"


def _dump_error(resp: requests.Response) -> str:
    rid = resp.headers.get("x-request-id", "<none>")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return f"HTTP {resp.status_code} (request-id: {rid})\n{body}"


def create_video(
    *,
    prompt: str,
    size: str,
    seconds: int,
    model: str,
    input_reference: Path | None,
    api_key: str,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """
    Always send multipart/form-data (compatible with /videos and input_reference uploads).
    """
    headers_auth = {"Authorization": f"Bearer {api_key}"}

    with ExitStack() as stack:
        files: dict[str, Any] = {
            "model": (None, model),
            "prompt": (None, prompt),
            "seconds": (None, str(int(seconds))),
        }
        if size:
            files["size"] = (None, size)

        if input_reference is not None:
            ref = Path(input_reference)
            f = stack.enter_context(open(ref, "rb"))
            files["input_reference"] = (ref.name, f, guess_mime(ref))

        r = requests.post(
            f"{API_BASE}/videos",
            headers=headers_auth,
            files=files,
            timeout=timeout_sec,
        )
        if r.status_code >= 400:
            raise RuntimeError("Create video failed:\n" + _dump_error(r))
        return r.json()


def retrieve_video(*, video_id: str, api_key: str, timeout_sec: int = 60) -> dict[str, Any]:
    headers_auth = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(f"{API_BASE}/videos/{video_id}", headers=headers_auth, timeout=timeout_sec)
    if r.status_code >= 400:
        raise RuntimeError("Retrieve video failed:\n" + _dump_error(r))
    return r.json()


def download_video_content(
    *,
    video_id: str,
    out_path: Path,
    api_key: str,
    variant: str = "video",
    timeout_sec: int = 600,
) -> Path:
    headers_auth = {"Authorization": f"Bearer {api_key}"}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(
        f"{API_BASE}/videos/{video_id}/content",
        headers=headers_auth,
        params={"variant": variant},
        stream=True,
        timeout=timeout_sec,
    ) as r:
        if r.status_code >= 400:
            raise RuntimeError("Download failed:\n" + _dump_error(r))
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path


def poll_until_complete(
    *,
    job: dict[str, Any],
    api_key: str,
    poll_interval_sec: float = 2.0,
    print_progress_bar: bool = True,
) -> dict[str, Any]:
    video = dict(job)
    vid = video["id"]

    def bar(pct: float, width: int = 30) -> str:
        pct = max(0.0, min(100.0, pct))
        filled = int(pct / 100 * width)
        return "=" * filled + "-" * (width - filled)

    while video.get("status") in ("queued", "in_progress"):
        if print_progress_bar:
            pct = float(video.get("progress", 0) or 0)
            status_text = "Queued" if video["status"] == "queued" else "Processing"
            print(f"\r{status_text}: [{bar(pct)}] {pct:5.1f}%", end="", flush=True)
        time.sleep(poll_interval_sec)
        video = retrieve_video(video_id=vid, api_key=api_key)

    if print_progress_bar:
        print(flush=True)

    if video.get("status") != "completed":
        msg = (video.get("error") or {}).get("message", f"Job {vid} failed")
        raise RuntimeError(msg)

    return video


def extract_last_frame(video_path: Path, out_image_path: Path) -> Path:
    import cv2  # imported here so dependency checks are centralized

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    success = False
    frame = None

    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        success, frame = cap.read()

    if not success or frame is None:
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frame = f
            success = True

    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"Could not read last frame from {video_path}")

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_image_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write {out_image_path}")

    return out_image_path


# ----------------------------
# 4) Planner (Responses API)
# ----------------------------

def _extract_text_from_responses_obj(resp: Any) -> str:
    """
    Best-effort extraction across SDK versions.
    Prefer resp.output_text when available.
    """
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t

    # Attempt to traverse resp.output content blocks if present.
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks: list[str] = []
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str):
                        chunks.append(txt)
        if chunks:
            return "\n".join(chunks)

    # Last resort: serialize
    try:
        if hasattr(resp, "to_dict"):
            return json.dumps(resp.to_dict())
    except Exception:
        pass
    return ""


def plan_prompts_with_ai(
    *,
    base_prompt: str,
    seconds_per_segment: int,
    num_generations: int,
    planner_model: str,
) -> list[dict[str, Any]]:
    """
    Returns a list of segments:
      [{"title": "...", "seconds": <int>, "prompt": "<full prompt block>"}, ...]
    """
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY

    user_input = f"""
BASE PROMPT: {base_prompt}

GENERATION LENGTH (seconds): {seconds_per_segment}
TOTAL GENERATIONS: {num_generations}

Return exactly {num_generations} segments.
""".strip()

    resp = client.responses.create(
        model=planner_model,
        instructions=PLANNER_SYSTEM_INSTRUCTIONS,
        input=user_input,
    )

    text = _extract_text_from_responses_obj(resp)
    if not text.strip():
        raise RuntimeError("Planner returned no text; try changing --planner-model.")

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("Planner did not return JSON. Inspect response and adjust instructions.")

    data = json.loads(m.group(0))
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Planner JSON missing 'segments' list.")

    # Clamp to requested length
    if len(segments) != num_generations:
        segments = segments[:num_generations]

    # Enforce durations
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError("Planner segments must be objects.")
        seg["seconds"] = int(seconds_per_segment)

    return segments


# ----------------------------
# 5) Generation + concatenation
# ----------------------------

def chain_generate_sora(
    *,
    segments: list[dict[str, Any]],
    size: str,
    model: str,
    out_dir: Path,
    api_key: str,
    poll_interval_sec: float,
    print_progress_bar: bool,
) -> list[Path]:
    input_ref: Path | None = None
    segment_paths: list[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments, start=1):
        secs = int(seg["seconds"])
        prompt = str(seg["prompt"])

        print(f"\n=== Generating Segment {i}/{len(segments)} — {secs}s ===")
        job = create_video(
            prompt=prompt,
            size=size,
            seconds=secs,
            model=model,
            input_reference=input_ref,
            api_key=api_key,
        )
        print("Started job:", job.get("id"), "| status:", job.get("status"))

        completed = poll_until_complete(
            job=job,
            api_key=api_key,
            poll_interval_sec=poll_interval_sec,
            print_progress_bar=print_progress_bar,
        )

        seg_path = out_dir / f"segment_{i:02d}.mp4"
        download_video_content(video_id=completed["id"], out_path=seg_path, api_key=api_key, variant="video")
        print("Saved", seg_path)
        segment_paths.append(seg_path)

        frame_path = out_dir / f"segment_{i:02d}_last.jpg"
        extract_last_frame(seg_path, frame_path)
        print("Extracted last frame ->", frame_path)
        input_ref = frame_path

    return segment_paths


def _ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or install imageio-ffmpeg.")
    return ff


def concatenate_segments(segment_paths: list[Path], out_path: Path) -> Path:
    """
    Prefer MoviePy if available; otherwise use ffmpeg concat demuxer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    moviepy_ok = importlib.util.find_spec("moviepy") is not None
    if moviepy_ok:
        from moviepy import VideoFileClip, concatenate_videoclips

        clips = [VideoFileClip(str(p)) for p in segment_paths]
        try:
            target_fps = clips[0].fps or 24
            result = concatenate_videoclips(clips, method="compose")
            result.write_videofile(
                str(out_path),
                codec="libx264",
                audio_codec="aac",
                fps=target_fps,
                preset="medium",
                threads=0,
            )
        finally:
            for c in clips:
                try:
                    c.close()
                except Exception:
                    pass
        return out_path

    # ffmpeg fallback
    ffmpeg = _ffmpeg_bin()
    concat_list = out_path.with_suffix(".concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in segment_paths:
            # concat demuxer requires: file '/path/to/file'
            f.write(f"file '{p.as_posix()}'\n")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    return out_path


# ----------------------------
# 6) CLI / main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan -> generate -> chain -> concatenate Sora video segments.")
    p.add_argument("--base-prompt", type=str, default="Gameplay footage of a game releasing in 2027, a car driving through a futuristic city")
    p.add_argument("--seconds-per-segment", type=int, default=8, choices=(4, 8, 12))
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--planner-model", type=str, default=os.environ.get("PLANNER_MODEL", "gpt-5.2"))
    p.add_argument("--sora-model", type=str, default=os.environ.get("SORA_MODEL", "sora-2"))
    p.add_argument("--size", type=str, default=os.environ.get("SIZE", "1280x720"))
    p.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "sora_ai_planned_chain")))
    p.add_argument("--poll-interval-sec", type=float, default=float(os.environ.get("POLL_INTERVAL_SEC", "2")))
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--install", action="store_true", help="Install missing dependencies into the current interpreter.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_deps(install=bool(args.install))

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    print_progress_bar = not bool(args.no_progress)

    segments_plan = plan_prompts_with_ai(
        base_prompt=args.base_prompt,
        seconds_per_segment=args.seconds_per_segment,
        num_generations=args.num_generations,
        planner_model=args.planner_model,
    )

    print("AI-planned segments:\n")
    for i, seg in enumerate(segments_plan, start=1):
        print(f"[{i:02d}] {seg['seconds']}s — {seg.get('title', '(untitled)')}")
        print(seg.get("prompt", ""))
        print("-" * 80)

    segment_paths = chain_generate_sora(
        segments=segments_plan,
        size=args.size,
        model=args.sora_model,
        out_dir=args.out_dir,
        api_key=api_key,
        poll_interval_sec=args.poll_interval_sec,
        print_progress_bar=print_progress_bar,
    )

    final_path = args.out_dir / "combined.mp4"
    concatenate_segments(segment_paths, final_path)
    print("\nWrote combined video:", final_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())