import re
from pathlib import Path

import yaml
from slugify import slugify
from tqdm import tqdm


TRANSCRIPTS_DIR = Path(
    "/media/sf_D_DRIVE/YandexDisk/Obsidian Vault/Expeditions/transcripts"
)


def safe_name(text: str) -> str:
    value = slugify(text, separator="-", lowercase=True)
    return value[:90] or "audio"


def read_frontmatter(md_path: Path):
    raw = md_path.read_text(encoding="utf-8")

    if not raw.startswith("---"):
        return None, raw

    parts = raw.split("---", 2)

    if len(parts) < 3:
        return None, raw

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")

    return metadata, body


def write_frontmatter(md_path: Path, metadata: dict, body: str):
    lines = ["---"]

    for k, v in metadata.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f'  - "{item}"')
        else:
            lines.append(f'{k}: "{v}"')

    lines.append("---")
    lines.append("")
    lines.append(body)

    md_path.write_text("\n".join(lines), encoding="utf-8")


def yandex_relative(path: Path):
    parts = path.parts

    for marker in ["YandexDisk", "Yandex.Disk"]:
        if marker in parts:
            i = parts.index(marker)
            return str(Path(*parts[i:]))

    return str(path)


def rename_source(md_path: Path):
    metadata, body = read_frontmatter(md_path)

    if not metadata:
        return

    source_file = metadata.get("source_file")

    if not source_file:
        return

    src = Path(source_file)

    if not src.exists():
        print(f"Нет файла: {src}")
        return

    title = metadata.get("title", "audio")
    date = metadata.get("date", "unknown-date")
    sha = str(metadata.get("sha256", ""))[:8]

    ext = src.suffix.lower()

    new_name = f"{date}_{safe_name(title)}_{sha}{ext}"
    new_path = src.with_name(new_name)

    if src == new_path:
        return

    if not new_path.exists():
        src.rename(new_path)

    metadata["source_file"] = str(new_path)
    metadata["yandex_path"] = yandex_relative(new_path)

    write_frontmatter(md_path, metadata, body)

    print(f"\nOK")
    print(f"{src.name}")
    print(" -> ")
    print(f"{new_path.name}")


def main():
    files = sorted(TRANSCRIPTS_DIR.glob("*.md"))

    print(f"Найдено md: {len(files)}")

    for md in tqdm(files):
        try:
            rename_source(md)
        except Exception as e:
            print(f"\nОшибка: {md}")
            print(e)


if __name__ == "__main__":
    main()
