#!/usr/bin/env python3
"""Create the lightweight MP3 attachments missing from a reprocessing queue.

The queue state is treated as the gate: a note is touched only after its worker
records it in ``completed``. This lets the repair process safely watch a live
queue without racing the transcription worker.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


FILTER = "afftdn=nr=6:nf=-40:tn=1"
HEADING = "## Аудио для ручной проверки"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def add_embed(text, relative_audio):
    embed = f"{HEADING}\n\n![[{relative_audio}]]\n\nОблегчённая копия: MP3 64 кбит/с, моно, мягкое шумоподавление. Оригинал и ссылки на него сохранены."
    if HEADING in text:
        start = text.index(HEADING)
        next_heading = text.find("\n## ", start + len(HEADING))
        end = len(text) if next_heading < 0 else next_heading + 1
        old_section = text[start:end].rstrip()
        if f"![[{relative_audio}]]" in old_section:
            return text, False
        replacement = embed + "\n\n"
        return text[:start] + replacement + text[end:], True

    transcript = "## Расшифровка"
    position = text.find(transcript)
    if position < 0:
        raise ValueError("Заголовок ## Расшифровка не найден")
    return text[:position] + embed + "\n\n" + text[position:], True


def repair_one(entry, run_dir, preview_dir, record):
    note = Path(entry["note"])
    source = Path(entry["source"])
    expected_source_hash = entry["sha256"]
    expected_seconds = float(entry["duration_seconds"])
    if not note.is_file():
        raise FileNotFoundError(f"Заметка отсутствует: {note}")
    if not source.is_file() or digest(source) != expected_source_hash:
        raise RuntimeError("Исходное аудио отсутствует или его SHA-256 не совпадает")

    raw = note.read_bytes()
    original_note_hash = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    audio_name = f"{note.stem}_mild_20261103.mp3"
    preview = preview_dir / audio_name
    relative_audio = f"Expeditions/audio_preview/{audio_name}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not preview.is_file():
        part = preview.with_name(preview.stem + ".part.mp3")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(source), "-vn", "-af", FILTER, "-ac", "1", "-ar", "16000",
                    "-c:a", "libmp3lame", "-b:a", "64k", str(part),
                ],
                check=True,
            )
            actual_seconds = duration(part)
            if abs(actual_seconds - expected_seconds) > max(2.0, expected_seconds * 0.005):
                raise RuntimeError(f"Длительность MP3 отличается: {actual_seconds:.2f} с")
            if digest(source) != expected_source_hash:
                raise RuntimeError("Исходное аудио изменилось во время создания MP3")
            part.replace(preview)
        finally:
            if part.exists():
                part.unlink()

    updated, changed = add_embed(text, relative_audio)
    if changed:
        backup = run_dir / "audio-preview-backups" / note.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(raw)
        elif backup.read_bytes() != raw:
            raise RuntimeError(f"Резервная копия уже существует и отличается: {backup}")
        if digest(note) != original_note_hash:
            raise RuntimeError("Заметка изменилась во время создания MP3; embed не добавлен")
        temporary_note = note.with_name(note.name + ".audio-preview.tmp")
        try:
            with temporary_note.open("x", encoding="utf-8") as stream:
                stream.write(updated)
            temporary_note.replace(note)
        finally:
            if temporary_note.exists():
                temporary_note.unlink()

    record[str(note)] = {
        "status": "repaired",
        "mp3": str(preview),
        "embedded": True,
        "note_changed": changed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--repair-state", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Wait for future queue completions")
    parser.add_argument("--interval", type=int, default=45)
    arguments = parser.parse_args()

    entries = json.loads(arguments.manifest.read_text(encoding="utf-8"))["notes"]
    repair = {"status": "running", "repaired": {}, "failed": {}}
    save_json(arguments.repair_state, repair)

    while True:
        queue = json.loads(arguments.queue_state.read_text(encoding="utf-8"))
        completed = set(queue.get("completed", []))
        for entry in entries:
            key = entry["note"]
            if key not in completed or key in repair["repaired"] or key in repair["failed"]:
                continue
            try:
                repair_one(entry, arguments.repair_state.parent, arguments.preview_dir, repair["repaired"])
                print(f"MP3 готов: {Path(key).name}", flush=True)
            except Exception as error:  # persist a per-note failure and continue the queue
                repair["failed"][key] = f"{type(error).__name__}: {error}"
                print(f"Ошибка MP3: {Path(key).name}: {error}", flush=True)
            save_json(arguments.repair_state, repair)

        if not arguments.watch or queue.get("status") in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(max(arguments.interval, 5))

    repair["status"] = "completed_with_errors" if repair["failed"] else "completed"
    repair["queue_status"] = queue.get("status")
    save_json(arguments.repair_state, repair)
    print(json.dumps({"status": repair["status"], "repaired": len(repair["repaired"]), "failed": repair["failed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

