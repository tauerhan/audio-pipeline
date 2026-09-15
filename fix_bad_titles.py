import re
from pathlib import Path

import yaml
from slugify import slugify
from tqdm import tqdm


CONFIG_PATH = Path("config.yaml")


BAD_WORDS = {
    "oexperience",
    "audio",
    "note",
    "transcript",
    "summary",
    "000",
    "001",
    "002",
    "003",
}


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def safe_name(text: str):
    return slugify(text, separator="-", lowercase=True)[:90] or "audio"


def yandex_relative(path: Path):
    parts = path.parts

    for marker in ["YandexDisk", "Yandex.Disk"]:
        if marker in parts:
            i = parts.index(marker)
            return str(Path(*parts[i:]))

    return str(path)


def clean_title_from_filename(path: Path):
    name = path.stem

    name = re.sub(r"^\d{4}-\d{2}-\d{2}[_\s-]*", "", name)
    name = re.sub(r"_[a-f0-9]{8}$", "", name, flags=re.IGNORECASE)

    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()

    return name.title() or "Audio"


def is_bad_name(path: Path):
    stem = path.stem.lower()

    for word in BAD_WORDS:
        if word in stem:
            return True

    if re.search(r"_00\d_[a-f0-9]{8}$", stem):
        return True

    return False


def replace_h1(body: str, title: str):
    lines = body.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i] = f"# {title}"
            return "\n".join(lines) + "\n"

    return f"# {title}\n\n" + body


def rename_pair(md_path: Path):
    metadata, body = read_note(md_path)

    if not metadata:
        return False

    sha = str(metadata.get("sha256", ""))[:8]
    date = str(metadata.get("date", "unknown-date"))
    title = str(metadata.get("title") or clean_title_from_filename(md_path))

    source_file = metadata.get("source_file")
    source_path = Path(source_file) if source_file else None

    should_fix_md = is_bad_name(md_path)
    should_fix_source = source_path and source_path.exists() and is_bad_name(source_path)

    if not should_fix_md and not should_fix_source:
        return False

    new_base = f"{date}_{safe_name(title)}_{sha}"

    if source_path and source_path.exists():
        new_source = source_path.with_name(new_base + source_path.suffix.lower())

        if new_source != source_path:
            if not new_source.exists():
                source_path.rename(new_source)
            source_path = new_source

        metadata["source_file"] = str(source_path)
        metadata["yandex_path"] = yandex_relative(source_path)

    new_md = md_path.with_name(new_base + ".md")

    metadata["title"] = title
    body = replace_h1(body, title)

    write_note(md_path, metadata, body)

    if new_md != md_path:
        if not new_md.exists():
            md_path.rename(new_md)
            md_path = new_md

    print(f"OK: {md_path.name}")
    return True


def main():
    cfg = load_config()
    transcripts_dir = Path(cfg["obsidian_dir"]) / "transcripts"

    files = sorted(transcripts_dir.glob("*.md"))

    fixed = 0

    for md in tqdm(files):
        if rename_pair(md):
            fixed += 1

    print(f"\nИсправлено: {fixed}")


if __name__ == "__main__":
    main()
