import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any

import requests
import yaml
from slugify import slugify
from tqdm import tqdm
from faster_whisper import WhisperModel


STATE_FILE = "pipeline_state.json"
INDEX_FILE = "pipeline_index.csv"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(state_path: Path):
    if not state_path.exists():
        return {"files": {}}

    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = state_path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    tmp.replace(state_path)


def quick_file_id(path: Path) -> str:
    st = path.stat()
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def hash_from_filename(path: Path) -> str | None:
    match = re.search(
        r"_([a-f0-9]{8})(?=\.[^.]+$)",
        path.name,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).lower()

    return None


def already_processed_by_hash(state: dict, short_hash: str) -> bool:
    for item in state["files"].values():
        sha = str(item.get("sha256", "")).lower()
        status = item.get("status")

        if sha.startswith(short_hash.lower()) and status in {"ok", "renamed"}:
            return True

    return False


def status_for_file(path: Path, state: dict) -> str:
    qid = quick_file_id(path)

    status = state["files"].get(qid, {}).get("status")

    if status:
        return status

    short_hash = hash_from_filename(path)

    if short_hash and already_processed_by_hash(state, short_hash):
        return "ok"

    return "new"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def detect_date(path: Path) -> str:
    name = path.stem.strip()

    match = re.match(r"^(\d{2})(\d{2})(\d{2})", name)

    if match:
        yy, mm, dd = match.groups()

        yy_int = int(yy)

        if yy_int >= 90:
            year = 1900 + yy_int
        else:
            year = 2000 + yy_int

        month = int(mm)
        day = int(dd)

        if 1900 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    return datetime.fromtimestamp(
        path.stat().st_mtime
    ).strftime("%Y-%m-%d")


def human_title_from_filename(path: Path) -> str:
    name = path.stem.strip()

    name = re.sub(r"^\d{6}[\s_-]*", "", name)
    name = re.sub(r"_[a-f0-9]{8}$", "", name, flags=re.IGNORECASE)

    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return path.stem

    return name.title()


def safe_title_text(text: str, fallback: str = "audio") -> str:
    title = slugify(text, separator="-", lowercase=True)[:90]
    return title or fallback


def sanitize_tag(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-zа-яё0-9_-]", "", text)
    return text


def clean_yaml_scalar(text: str) -> str:
    return str(text).replace('"', "'").replace("\n", " ").strip()


def yandex_relative_path(path: Path) -> str:
    parts = path.parts

    for marker in ["Yandex.Disk", "YandexDisk"]:
        if marker in parts:
            i = parts.index(marker)
            return str(Path(*parts[i:]))

    return str(path)


def find_audio_files(source_dir: Path, extensions):
    result = []

    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            result.append(p)

    return sorted(result, key=lambda x: str(x).lower())


def filter_files(files, include=None):
    if include:
        inc = include.lower()
        files = [f for f in files if inc in str(f).lower()]

    return files


def convert_to_wav(src: Path, dst: Path, sample_rate: int, channels: int):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-af", "highpass=f=80,lowpass=f=8000,dynaudnorm",
        str(dst),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
def transcribe(model, wav_path: Path, language: str, vad_filter: bool):
    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=(
            "Русская речь. Экспедиция, поездка, разговор, интервью. "
            "Возможен шум машины, ветер, дорога, музыка."
        ),
        prompt_reset_on_temperature=0.5,
    )

    rows = []
    full_text = []

    for seg in segments:
        text = seg.text.strip()

        if text:
            rows.append((seg.start, seg.end, text))
            full_text.append(text)

    return rows, "\n".join(full_text)

def extract_basic_tags(original: Path, full_text: str):
    tags = ["audio", "transcript"]

    combined = (
        f"{str(original).lower()} "
        f"{original.stem.lower()} "
        f"{full_text[:3000].lower()}"
    )

    keywords = {
        "expedition": [
            "экспеди",
            "поход",
            "перевал",
            "гора",
            "алтай",
            "кавказ",
            "памир",
            "ук-кок",
        ],
        "car": [
            "машин",
            "дорог",
            "авто",
        ],
        "interview": [
            "интервью",
            "вопрос",
            "ответ",
        ],
        "technique": [
            "техника",
            "практика",
            "упражнение",
        ],
        "spiritual": [
            "чакра",
            "энергия",
            "аура",
            "вселенная",
        ],
    }

    for tag, words in keywords.items():
        if any(w in combined for w in words):
            tags.append(tag)

    return sorted(set(tags))


def make_stub_ai(original: Path, full_text: str, basic_tags: list[str]) -> dict[str, Any]:
    title = human_title_from_filename(original)

    return {
        "title": title,
        "short_title": title[:90],
        "summary": "_AI summary ещё не создан._",
        "summary_status": "pending",
        "tags": basic_tags,
        "people": [],
        "places": [],
        "topics": [],
        "quality": "unknown",
    }


def call_ollama_ai(cfg: dict, original: Path, full_text: str, basic_tags: list[str]) -> dict[str, Any]:
    ai_cfg = cfg.get("ai", {})

    if not ai_cfg.get("enabled", False):
        return make_stub_ai(original, full_text, basic_tags)

    prompt = f"""
Ты анализируешь расшифровку аудио экспедиции.

Верни JSON:

{{
  "title": "...",
  "short_title": "...",
  "summary": "...",
  "tags": [],
  "people": [],
  "places": [],
  "topics": [],
  "quality": "good|medium|bad"
}}

Текст:
{full_text[:12000]}
"""

    try:
        response = requests.post(
            ai_cfg.get("url", "http://localhost:11434/api/generate"),
            json={
                "model": ai_cfg.get("model", "qwen2.5:7b-instruct"),
                "prompt": prompt,
                "stream": False,
            },
            timeout=240,
        )

        data = response.json()

        raw = data.get("response", "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return make_stub_ai(original, full_text, basic_tags)

        parsed = json.loads(match.group(0))

        parsed["tags"] = sorted(
            set(basic_tags + parsed.get("tags", []))
        )

        parsed["summary_status"] = "ready"

        return parsed

    except Exception:
        return make_stub_ai(original, full_text, basic_tags)


def rename_source_file(src: Path, date_prefix: str, short_title: str, digest: str) -> Path:
    ext = src.suffix.lower()

    new_name = (
        f"{date_prefix}_"
        f"{safe_title_text(short_title)}_"
        f"{digest[:8]}"
        f"{ext}"
    )

    new_path = src.with_name(new_name)

    if src == new_path:
        return src

    if new_path.exists():
        return new_path

    src.rename(new_path)

    return new_path


def write_markdown(
    md_path: Path,
    source_file: Path,
    rows,
    full_text,
    digest: str,
    ai: dict,
):
    md_path.parent.mkdir(parents=True, exist_ok=True)

    created = detect_date(source_file)

    title = ai.get("title") or human_title_from_filename(source_file)

    tags = ai.get("tags", [])

    yandex_path = yandex_relative_path(source_file)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("---\n")

        f.write(f'title: "{clean_yaml_scalar(title)}"\n')
        f.write(f"date: {created}\n")

        f.write(f'source_file: "{source_file}"\n')
        f.write(f'yandex_path: "{yandex_path}"\n')

        f.write(f'sha256: "{digest}"\n')

        f.write(
            f'summary_status: "{ai.get("summary_status", "pending")}"\n'
        )

        f.write(
            f'recognition_quality: "{clean_yaml_scalar(ai.get("quality", "unknown"))}"\n'
        )

        f.write("tags:\n")

        for tag in tags:
            f.write(f"  - {sanitize_tag(tag)}\n")

        f.write("---\n\n")

        f.write(f"# {title}\n\n")

        f.write("## Summary\n\n")
        f.write(ai.get("summary", "_AI summary ещё не создан._"))
        f.write("\n\n")

        f.write("## Исходный файл\n\n")
        f.write(f"`{source_file}`\n\n")

        f.write("## Яндекс.Диск\n\n")
        f.write(f"`{yandex_path}`\n\n")

        f.write("## Расшифровка\n\n")

        if full_text:
            f.write(full_text)
        else:
            f.write("_Речь не распознана._")

        f.write("\n\n")

        f.write("## С таймкодами\n\n")

        for start, end, text in rows:
            f.write(f"- [{start:0.1f}–{end:0.1f}] {text}\n")


def append_index(index_path: Path, row: dict):
    exists = index_path.exists()

    with index_path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sha256",
            "source_file",
            "source_date",
            "title",
            "tags",
            "markdown_file",
            "status",
            "processed_at",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def list_candidates(files, state, max_rows=200):
    print("\nКандидаты на обработку:\n")

    count = 0

    for f in files:
        status = status_for_file(f, state)

        print(f"[{status}] {detect_date(f)} | {f}")

        count += 1

        if count >= max_rows:
            print(f"... показано {max_rows}, всего {len(files)}")
            break


def cleanup_wav(wav_path: Path, keep_processed_audio: bool):
    if keep_processed_audio:
        return

    if wav_path.exists():
        wav_path.unlink()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-wav", action="store_true")

    args = parser.parse_args()

    cfg = load_config()

    source_dir = Path(cfg["source_dir"])
    obsidian_dir = Path(cfg["obsidian_dir"])

    processed_audio_dir = obsidian_dir / "processed_audio"
    transcripts_dir = obsidian_dir / "transcripts"

    index_path = obsidian_dir / INDEX_FILE
    state_path = obsidian_dir / STATE_FILE

    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)

    audio_files = find_audio_files(
        source_dir,
        [e.lower() for e in cfg["audio_extensions"]],
    )

    audio_files = filter_files(
        audio_files,
        include=args.include,
    )

    print(f"Найдено аудиофайлов: {len(audio_files)}")

    if args.list:
        list_candidates(audio_files, state)
        return

    files_to_process = []

    for src in audio_files:
        qid = quick_file_id(src)

        if not args.force:
            current_status = state["files"].get(qid, {}).get("status")

            if current_status == "ok":
                continue

            short_hash = hash_from_filename(src)

            if short_hash and already_processed_by_hash(state, short_hash):
                continue

        files_to_process.append(src)

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    print(f"К обработке: {len(files_to_process)}")

    if args.dry_run:
        for f in files_to_process:
            print(f"{detect_date(f)} | {f}")
        return

    if not files_to_process:
        print("Новых файлов нет.")
        return

    model = WhisperModel(
        cfg["model"],
        device=cfg["transcription"]["device"],
        compute_type=cfg["transcription"]["compute_type"],
    )

    for src in tqdm(files_to_process):
        original_qid = quick_file_id(src)

        try:
            digest = file_hash(src)

            date_prefix = detect_date(src)

            clean_title = human_title_from_filename(src)

            tmp_name = (
                f"{date_prefix}_"
                f"{safe_title_text(clean_title)}_"
                f"{digest[:8]}"
            )

            wav_path = processed_audio_dir / f"{tmp_name}.wav"

            convert_to_wav(
                src,
                wav_path,
                cfg["convert"]["sample_rate"],
                cfg["convert"]["channels"],
            )

            rows, full_text = transcribe(
                model,
                wav_path,
                cfg["language"],
                cfg["transcription"]["vad_filter"],
            )

            basic_tags = extract_basic_tags(src, full_text)

            ai = call_ollama_ai(
                cfg,
                src,
                full_text,
                basic_tags,
            )

            short_title = ai.get("short_title") or clean_title

            new_source_path = rename_source_file(
                src,
                date_prefix,
                short_title,
                digest,
            )

            md_base = (
                f"{date_prefix}_"
                f"{safe_title_text(short_title)}_"
                f"{digest[:8]}"
            )

            md_path = transcripts_dir / f"{md_base}.md"

            write_markdown(
                md_path=md_path,
                source_file=new_source_path,
                rows=rows,
                full_text=full_text,
                digest=digest,
                ai=ai,
            )

            cleanup_wav(
                wav_path,
                keep_processed_audio=args.keep_wav,
            )

            append_index(
                index_path,
                {
                    "sha256": digest,
                    "source_file": str(new_source_path),
                    "source_date": date_prefix,
                    "title": ai.get("title", ""),
                    "tags": ";".join(ai.get("tags", [])),
                    "markdown_file": str(md_path),
                    "status": "ok",
                    "processed_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
            )

            new_qid = quick_file_id(new_source_path)

            state["files"][original_qid] = {
                "status": "renamed",
                "new_path": str(new_source_path),
                "sha256": digest,
            }

            state["files"][new_qid] = {
                "status": "ok",
                "sha256": digest,
                "source_file": str(new_source_path),
                "markdown_file": str(md_path),
            }

            save_state(state_path, state)

        except Exception as e:
            state["files"][original_qid] = {
                "status": "error",
                "source_file": str(src),
                "error": str(e),
            }

            save_state(state_path, state)

            print(f"\nОшибка: {src}")
            print(e)


if __name__ == "__main__":
    main()
