import hashlib
import re
import subprocess
from pathlib import Path

import yaml
from faster_whisper import WhisperModel
from tqdm import tqdm


CONFIG_PATH = Path("config.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def yandex_relative(path: Path):
    parts = path.parts

    for marker in ["YandexDisk", "Yandex.Disk"]:
        if marker in parts:
            i = parts.index(marker)
            return str(Path(*parts[i:]))

    return str(path)


def find_source_by_hash(source_dir: Path, wanted_sha: str, extensions: list[str]) -> Path | None:
    short_sha = wanted_sha[:8].lower()

    # Быстрый поиск по hash в имени
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in extensions:
            continue

        if short_sha in p.name.lower():
            return p

    # Медленный поиск по полному hash
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in extensions:
            continue

        try:
            if sha256_file(p) == wanted_sha:
                return p
        except Exception:
            continue

    return None


def read_note(path: Path):
    raw = path.read_text(encoding="utf-8")

    if not raw.startswith("---"):
        return None, raw

    parts = raw.split("---", 2)

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")

    return metadata, body


def write_note(path: Path, metadata: dict, body: str):
    lines = ["---"]

    order = [
        "title",
        "date",
        "source_file",
        "yandex_path",
        "processed_audio",
        "sha256",
        "summary_status",
        "recognition_quality",
        "tags",
        "people",
        "places",
        "topics",
    ]

    written = set()

    for key in order:
        if key not in metadata:
            continue

        written.add(key)
        value = metadata[key]

        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f'  - "{item}"')
        elif key == "summary_status":
            lines.append(f"{key}: {value}")
        else:
            lines.append(f'{key}: "{value}"')

    for key, value in metadata.items():
        if key in written:
            continue

        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f'  - "{item}"')
        else:
            lines.append(f'{key}: "{value}"')

    lines.append("---")
    lines.append("")
    lines.append(body)

    path.write_text("\n".join(lines), encoding="utf-8")


def replace_section(body: str, section_name: str, content: str):
    pattern = re.compile(
        rf"(^## {re.escape(section_name)}\s*$)(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )

    replacement = f"## {section_name}\n\n{content.strip()}\n\n"

    if pattern.search(body):
        return pattern.sub(replacement, body)

    return body.rstrip() + "\n\n" + replacement


def convert_with_noise_reduction(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af",
        (
            "highpass=f=100,"
            "lowpass=f=7000,"
            "afftdn=nf=-25,"
            "dynaudnorm=f=150:g=15,"
            "loudnorm"
        ),
        str(dst),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def transcribe_retry(model, wav_path: Path):
    segments, info = model.transcribe(
        str(wav_path),
        language="ru",
        vad_filter=False,
        beam_size=8,
        best_of=5,
        temperature=0,
        condition_on_previous_text=False,
        initial_prompt=(
            "Русская речь. Экспедиционная аудиозапись. "
            "Возможны шум, ветер, разговоры на фоне."
        ),
    )

    rows = []
    text_parts = []

    for seg in segments:
        text = seg.text.strip()

        if text:
            rows.append((seg.start, seg.end, text))
            text_parts.append(text)

    return rows, "\n".join(text_parts)


def format_timestamps(rows):
    lines = []

    for start, end, text in rows:
        lines.append(f"- [{start:0.1f}–{end:0.1f}] {text}")

    return "\n".join(lines)


def is_failed(metadata: dict):
    return str(metadata.get("summary_status", "")).lower() == "skipped_no_transcript"


def main():
    cfg = load_config()

    source_dir = Path(cfg["source_dir"])
    obsidian_dir = Path(cfg["obsidian_dir"])

    extensions = [e.lower() for e in cfg["audio_extensions"]]

    transcripts_dir = obsidian_dir / "transcripts"
    retry_dir = obsidian_dir / "retry_audio"

    notes = sorted(transcripts_dir.glob("*.md"))

    failed_notes = []

    for note in notes:
        metadata, body = read_note(note)

        if metadata and is_failed(metadata):
            failed_notes.append(note)

    print(f"Проблемных заметок: {len(failed_notes)}")

    if not failed_notes:
        return

    model_name = cfg.get("retry_model", "large-v3")

    print(f"Модель retry: {model_name}")

    model = WhisperModel(
        model_name,
        device=cfg["transcription"]["device"],
        compute_type=cfg["transcription"]["compute_type"],
    )

    recovered = 0
    still_failed = 0

    for note in tqdm(failed_notes):
        metadata, body = read_note(note)

        source = Path(str(metadata.get("source_file", "")))

        if not source.exists():
            sha = str(metadata.get("sha256", "")).strip()

            source = find_source_by_hash(
                source_dir,
                sha,
                extensions,
            )

            if not source:
                print(f"\nНет исходника: {metadata.get('source_file')}")
                print(f"Не найден по sha256: {sha[:8]}")
                still_failed += 1
                continue

            metadata["source_file"] = str(source)
            metadata["yandex_path"] = yandex_relative(source)

        sha = str(metadata.get("sha256", ""))[:8]

        wav_path = retry_dir / f"{note.stem}_{sha}_retry.wav"

        try:
            convert_with_noise_reduction(source, wav_path)

            rows, full_text = transcribe_retry(model, wav_path)

            if len(full_text.strip()) < 50 or len(rows) < 2:
                metadata["summary_status"] = "skipped_no_transcript"
                metadata["recognition_quality"] = "failed"

                still_failed += 1

            else:
                metadata["summary_status"] = "pending"
                metadata["recognition_quality"] = "retry_recovered"

                tags = set(metadata.get("tags") or [])

                tags.discard("no-transcript")
                tags.add("retry-recovered")

                metadata["tags"] = sorted(tags)

                body = replace_section(
                    body,
                    "Summary",
                    "_AI summary ещё не создан после retry._",
                )

                body = replace_section(
                    body,
                    "Расшифровка",
                    full_text,
                )

                body = replace_section(
                    body,
                    "С таймкодами",
                    format_timestamps(rows),
                )

                recovered += 1

                print(f"\nRECOVERED: {note.name}")

            write_note(note, metadata, body)

        except Exception as e:
            print(f"\nОшибка retry: {note}")
            print(e)

            still_failed += 1

        finally:
            if wav_path.exists():
                wav_path.unlink()

    print("\nГотово")
    print(f"Восстановлено: {recovered}")
    print(f"Осталось проблемных: {still_failed}")


if __name__ == "__main__":
    main()
