"""
Direct HTTPS model downloader for faster-whisper models with resume support.
"""

import os
import sys
import time
import urllib.request
from typing import Optional

BASE_HF_URL = "https://huggingface.co/Systran/faster-whisper-{model}/resolve/main/{file}"

REQUIRED_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
]


def download_file_direct(url: str, dest_path: str, filename: str) -> None:
    """Download a file with 512KB chunk buffer and HTTP Range resume support."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"

    downloaded = 0
    if os.path.isfile(temp_path):
        downloaded = os.path.getsize(temp_path)

    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_length = int(resp.headers.get("Content-Length", 0))
            if status == 206:  # Partial Content
                total_size = downloaded + content_length
            else:
                total_size = content_length
                downloaded = 0
                mode = "wb"

            mode = "ab" if downloaded > 0 and status == 206 else "wb"
            start_time = time.time()
            last_print = 0

            with open(temp_path, mode) as out:
                while True:
                    chunk = resp.read(512 * 1024)  # 512 KB chunks!
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print > 0.4:
                        last_print = now
                        elapsed = max(0.1, now - start_time)
                        speed_mb = (downloaded / (1024 * 1024)) / elapsed
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        pct = (downloaded / total_size) * 100.0 if total_size else 0.0
                        sys.stdout.write(
                            f"\r  [Downloading {filename}] {pct:5.1f}% ({dl_mb:5.1f}/{tot_mb:5.1f} MB) at {speed_mb:4.1f} MB/s"
                        )
                        sys.stdout.flush()

        sys.stdout.write(f"\r  [Downloaded {filename}] 100.0% Complete!                             \n")
        sys.stdout.flush()

        if os.path.isfile(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)

    except Exception as e:
        sys.stdout.write(f"\nDownload error for {filename}: {e}\n")
        sys.stdout.flush()
        raise e


def ensure_model_downloaded(model_name: str = "base", target_dir: Optional[str] = None) -> str:
    """
    Ensure all model files exist in local directory.
    Returns the absolute path to the local model folder.
    """
    if os.path.isdir(model_name):
        return os.path.abspath(model_name)

    if not target_dir:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        target_dir = os.path.join(project_root, "models", model_name)

    os.makedirs(target_dir, exist_ok=True)

    all_exist = True
    for f in REQUIRED_FILES:
        fp = os.path.join(target_dir, f)
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            all_exist = False
            break

    if all_exist:
        return target_dir

    print(f"\nEnsuring Whisper model '{model_name}' is available locally in {target_dir}...")
    for fname in REQUIRED_FILES:
        dest_file = os.path.join(target_dir, fname)
        if os.path.isfile(dest_file) and os.path.getsize(dest_file) > 0:
            continue

        url = BASE_HF_URL.format(model=model_name, file=fname)
        download_file_direct(url, dest_file, fname)

    return target_dir


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    path = ensure_model_downloaded(name)
    print(f"Model ready at: {path}")
