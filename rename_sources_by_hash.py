import re
import hashlib
from pathlib import Path

import yaml
from slugify import slugify
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


def safe_name(text: str) -> str:
    return slugify(text, separator="-", lowercase=True)[:90] or "audio"


def read_note(md_path: Path):
    raw = md_path.read_text(encoding="utf-8")

    if not raw.startswith("---"):
        return None, raw

    parts = raw.split("---", 2)

    if len(parts) < 3:
        return None, raw

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")

    return metadata, body


def write_note(md_path: Path, metadata: dict, body: str):
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
        elif key in {"summary_status"}:
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

    md_path.write_text("\n".join(lines), encoding="utf-8")


def yandex_relative(path: Path) -> str:
    parts = path.parts

    for marker in ["YandexDisk", "Yandex.Disk"]:
        if marker in parts:
            i = parts.index(marker)
            return str(Path(*parts[i:]))

    return str(path)


def find_source_by_hash(source_dir: Path, wanted_sha: str, extensions: list[str]) -> Path | None:
    short_sha = wanted_sha[:8].lower()

    # Сначала быстро ищем по hash в имени
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in extensions:
            continue

        if short_sha in p.name.lower():
            return p

    # Потом медленно считаем sha256
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


def clean_title(metadata: dict, md_path: Path) -> str:
    title = metadata.get("title") or md_path.stem

    title = re.sub(r"^\d{4}-\d{2}-\d{2}[_\\s-]*", "", title)
    title = re.sub(r"_[a-f0-9]{8}$", "", title, flags=re.IGNORECASE)
    title = title.strip()

    return title or "audio"


def rename_source(source: Path, metadata: dict, md_path: Path) -> Path:
    date = str(metadata.get("date") or "unknown-date")
    sha = str(metadata.get("sha256") or "")[:8]

    title = clean_title(metadata, md_path)
    ext = source.suffix.lower()

    new_name = f"{date}_{safe_name(title)}_{sha}{ext}"
    new_path = source.with_name(new_name)

    if source == new_path:
        return source

    if new_path.exists():
        return new_path

    source.rename(new_path)

    return new_path


def main():
    cfg = load_config()

    source_dir = Path(cfg["source_dir"])
    obsidian_dir = Path(cfg["obsidian_dir"])
    transcripts_dir = obsidian_dir / "transcripts"

    extensions = [e.lower() for e in cfg["audio_extensions"]]

    md_files = sorted(transcripts_dir.glob("*.md"))

    print(f"Заметок найдено: {len(md_files)}")

    renamed = 0
    missing = 0
    already_ok = 0

    for md_path in tqdm(md_files):
        metadata, body = read_note(md_path)

        if not metadata:
            continue

        sha = str(metadata.get("sha256") or "").strip()

        if not sha:
            print(f"\nНет sha256: {md_path}")
            continue

        current_source = Path(str(metadata.get("source_file") or ""))

        source = None

        if current_source.exists():
            source = current_source
        else:
            source = find_source_by_hash(source_dir, sha, extensions)

        if not source:
            missing += 1
            print(f"\nНе найден исходник для: {md_path.name}")
            print(f"sha256: {sha[:8]}")
            continue

        new_source = rename_source(source, metadata, md_path)

        metadata["source_file"] = str(new_source)
        metadata["yandex_path"] = yandex_relative(new_source)
        metadata["processed_audio"] = ""

        write_note(md_path, metadata, body)

        if source == new_source:
            already_ok += 1
        else:
            renamed += 1
            print(f"\nOK rename:")
            print(f"{source.name}")
            print(f"-> {new_source.name}")

    print("\nГотово")
    print(f"Переименовано: {renamed}")
    print(f"Уже корректно: {already_ok}")
    print(f"Не найдено: {missing}")


if __name__ == "__main__":
    main()
