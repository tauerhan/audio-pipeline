import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests
import yaml
from slugify import slugify
from tqdm import tqdm


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_yaml_scalar(text: Any) -> str:
    return str(text).replace('"', "'").replace("\n", " ").strip()


def sanitize_tag(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-zа-яё0-9_-]", "", text)
    return text


def safe_filename(text: str, fallback: str = "note") -> str:
    value = slugify(text, separator="-", lowercase=True)[:90]
    return value or fallback


def is_bad_short_title(text: str) -> bool:
    value = text.strip().lower()
    return (
        len(value) < 4
        or value.isdigit()
        or value in {"000", "001", "002", "003", "note", "audio", "transcript", "summary"}
    )


def read_note(path: Path) -> tuple[dict, str, str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)

    if not match:
        return {}, raw, raw

    metadata = yaml.safe_load(match.group(1)) or {}
    body = raw[match.end():]
    return metadata, body, raw


def build_frontmatter(metadata: dict) -> str:
    lines = ["---"]

    preferred_order = [
        "title",
        "date",
        "source_file",
        "yandex_path",
        "processed_audio",
        "sha256",
        "summary_status",
        "recognition_quality",
    ]

    written = set()

    for key in preferred_order:
        if key not in metadata:
            continue

        value = metadata[key]
        written.add(key)

        if isinstance(value, str):
            if key == "summary_status":
                lines.append(f"{key}: {value}")
            else:
                lines.append(f'{key}: "{clean_yaml_scalar(value)}"')
        else:
            lines.append(f"{key}: {value}")

    for list_key in ["tags", "people", "places", "topics"]:
        if list_key in metadata:
            written.add(list_key)
            values = metadata.get(list_key) or []
            lines.append(f"{list_key}:")
            for item in values:
                if list_key == "tags":
                    item = sanitize_tag(item)
                    if item:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f'  - "{clean_yaml_scalar(item)}"')

    for key, value in metadata.items():
        if key in written:
            continue

        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f'  - "{clean_yaml_scalar(item)}"')
        elif isinstance(value, str):
            lines.append(f'{key}: "{clean_yaml_scalar(value)}"')
        else:
            lines.append(f"{key}: {value}")

    lines.append("---")
    return "\n".join(lines) + "\n\n"


def extract_section(body: str, section_name: str) -> str:
    matches = list(SECTION_RE.finditer(body))

    for i, match in enumerate(matches):
        title = match.group(1).strip().lower()
        if title == section_name.lower():
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            return body[start:end].strip()

    return ""


def replace_h1_title(body: str, title: str) -> str:
    lines = body.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i] = f"# {title}"
            return "\n".join(lines) + "\n"

    return f"# {title}\n\n" + body.lstrip()


def replace_or_insert_section(body: str, section_name: str, content: str) -> str:
    matches = list(SECTION_RE.finditer(body))

    for i, match in enumerate(matches):
        title = match.group(1).strip().lower()

        if title == section_name.lower():
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            new_section = f"## {section_name}\n\n{content.strip()}\n\n"
            return body[:start] + new_section + body[end:].lstrip("\n")

    first_section = SECTION_RE.search(body)

    if first_section:
        insert_at = first_section.start()
        return body[:insert_at].rstrip() + f"\n\n## {section_name}\n\n{content.strip()}\n\n" + body[insert_at:]

    return body.rstrip() + f"\n\n## {section_name}\n\n{content.strip()}\n"


def remove_sections(body: str, section_names: list[str]) -> str:
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return body

    ranges_to_remove = []
    names = {x.lower() for x in section_names}

    for i, match in enumerate(matches):
        title = match.group(1).strip().lower()
        if title in names:
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            ranges_to_remove.append((start, end))

    for start, end in reversed(ranges_to_remove):
        body = body[:start].rstrip() + "\n\n" + body[end:].lstrip()

    return body


def insert_after_summary(body: str, content: str) -> str:
    matches = list(SECTION_RE.finditer(body))

    for i, match in enumerate(matches):
        title = match.group(1).strip().lower()

        if title == "summary":
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            return body[:end].rstrip() + "\n\n" + content + body[end:].lstrip("\n")

    first_section = SECTION_RE.search(body)

    if first_section:
        return body[:first_section.start()].rstrip() + "\n\n" + content + body[first_section.start():]

    return content + body


def crop_text(text: str, limit: int) -> str:
    text = text.strip()

    if len(text) <= limit:
        return text

    head = text[: limit // 2]
    tail = text[-limit // 2:]

    return head + "\n\n...[середина расшифровки пропущена]...\n\n" + tail


def mark_no_transcript(note: Path, metadata: dict, body: str):
    metadata["summary_status"] = "skipped_no_transcript"
    metadata["recognition_quality"] = "failed"

    if "tags" not in metadata or metadata["tags"] is None:
        metadata["tags"] = []

    tags = set(metadata.get("tags") or [])
    tags.add("no-transcript")
    tags.add("problem-audio")
    metadata["tags"] = sorted(tags)

    body = replace_or_insert_section(
        body,
        "Summary",
        "_AI summary не создан: в заметке нет распознанной речи._",
    )

    new_raw = build_frontmatter(metadata) + body.lstrip("\n")
    note.write_text(new_raw, encoding="utf-8")


def call_ollama(cfg: dict, metadata: dict, transcript: str) -> dict:
    ai_cfg = cfg.get("ai", {})

    url = ai_cfg.get("ollama_url", DEFAULT_OLLAMA_URL)
    model = ai_cfg.get("summary_model") or ai_cfg.get("model", DEFAULT_MODEL)
    timeout = int(ai_cfg.get("summary_timeout", ai_cfg.get("timeout", 1800)))
    text_limit = int(ai_cfg.get("summary_text_limit", ai_cfg.get("text_limit", 5000)))
    num_ctx = int(ai_cfg.get("summary_num_ctx", ai_cfg.get("num_ctx", 4096)))

    source_transcript = transcript
    transcript = crop_text(transcript, text_limit)

    prompt = f"""
Ты архивариус русскоязычных аудиозаписей экспедиций, поездок, интервью и личных бесед.
Проанализируй расшифровку и верни строго JSON без markdown и без пояснений.

Текущее название: {metadata.get('title', '')}
Дата: {metadata.get('date', '')}
Путь Яндекс.Диска: {metadata.get('yandex_path', '')}
Исходный файл: {metadata.get('source_file', '')}

Задача:
1. Придумай понятное название заметки.
2. Сделай краткое содержание на русском языке.
3. Выдели темы, места, людей и теги для Obsidian.
4. Оцени качество расшифровки.

Правила:
- Не выдумывай людей и места, если их нет в тексте.
- Для поля people используй только точные фрагменты расшифровки, сохраняя исходный язык и написание; не переводи, например, «Учитель» в teacher.
- Если имя или обозначение человека не встречается в расшифровке в таком написании, не добавляй его в people.
- Теги должны быть короткими, без #, без пробелов, лучше через дефис.
- Summary должно быть 5-10 предложений.
- Не давай медицинских или мистических утверждений от себя; только описывай содержание записи.
- short_title не должен быть числом, например 000, 001, 002.
- short_title должен быть осмысленным названием из 3-7 слов.

Расшифровка:
{transcript}

Верни только JSON:
{{
  "title": "...",
  "short_title": "...",
  "summary": "...",
  "tags": ["..."],
  "people": ["..."],
  "places": ["..."],
  "topics": ["..."],
  "recognition_quality": "good|medium|poor"
}}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_ctx": num_ctx,
        },
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    raw = response.json().get("response", "{}").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Ollama returned non-JSON response: {raw[:1000]}")
        data = json.loads(match.group(0))

    return normalize_ai_result(data, metadata, source_transcript)


def retain_verbatim_people(values: Any, transcript: str) -> list[str]:
    if not isinstance(values, list):
        return []

    people = []
    seen = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in candidate.split()) + r"(?!\w)"
        match = re.search(pattern, transcript, flags=re.IGNORECASE)
        if match and match.group(0).casefold() not in seen:
            people.append(match.group(0))
            seen.add(match.group(0).casefold())
    return people


def normalize_ai_result(data: dict, metadata: dict, transcript: str) -> dict:
    title = clean_yaml_scalar(data.get("title") or metadata.get("title") or "Без названия")
    short_title = clean_yaml_scalar(data.get("short_title") or title)

    if is_bad_short_title(short_title):
        short_title = title

    summary = str(data.get("summary") or "_AI summary не создан._").strip()

    existing_tags = metadata.get("tags") or []
    ai_tags = data.get("tags") or []

    tags = sorted(
        set(
            [
                sanitize_tag(t)
                for t in list(existing_tags) + list(ai_tags)
                if sanitize_tag(t)
            ]
        )
    )

    people = [clean_yaml_scalar(x) for x in retain_verbatim_people(data.get("people"), transcript)]

    places = [
        clean_yaml_scalar(x)
        for x in data.get("places", [])
        if str(x).strip()
    ]

    topics = [
        clean_yaml_scalar(x)
        for x in data.get("topics", [])
        if str(x).strip()
    ]

    quality = str(
        data.get("recognition_quality")
        or data.get("quality")
        or metadata.get("recognition_quality")
        or "unknown"
    ).strip().lower()

    if quality not in {"good", "medium", "poor", "unknown"}:
        quality = "unknown"

    return {
        "title": title,
        "short_title": short_title,
        "summary": summary,
        "tags": tags,
        "people": people,
        "places": places,
        "topics": topics,
        "recognition_quality": quality,
    }


def markdown_list(title: str, items: list[str], wiki: bool = False) -> str:
    if not items:
        return ""

    lines = [f"## {title}", ""]

    for item in items:
        if wiki:
            lines.append(f"- [[{item}]]")
        else:
            lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n\n"


def update_note(path: Path, ai: dict, rename: bool = False) -> Path:
    metadata, body, raw = read_note(path)

    metadata["title"] = ai["title"]
    metadata["summary_status"] = "done"
    metadata["recognition_quality"] = ai["recognition_quality"]
    metadata["tags"] = ai["tags"]
    metadata["people"] = ai["people"]
    metadata["places"] = ai["places"]
    metadata["topics"] = ai["topics"]

    body = replace_h1_title(body, ai["title"])
    body = replace_or_insert_section(body, "Summary", ai["summary"])
    body = remove_sections(body, ["Темы", "Места", "Люди"])

    insert = ""
    insert += markdown_list("Темы", ai["topics"], wiki=False)
    insert += markdown_list("Места", ai["places"], wiki=True)
    insert += markdown_list("Люди", ai["people"], wiki=True)

    if insert.strip():
        body = insert_after_summary(body, insert.strip() + "\n\n")

    new_raw = build_frontmatter(metadata) + body.lstrip("\n")
    path.write_text(new_raw, encoding="utf-8")

    if rename:
        date = str(metadata.get("date", "")).strip() or "unknown-date"
        sha = str(metadata.get("sha256", ""))[:8]
        suffix = f"_{sha}" if sha else ""
        new_name = f"{date}_{safe_filename(ai['short_title'])}{suffix}.md"
        new_path = path.with_name(new_name)

        if new_path != path and not new_path.exists():
            path.rename(new_path)
            return new_path

    return path


def find_pending_notes(
    transcripts_dir: Path,
    include: str | None = None,
    force: bool = False,
) -> list[Path]:
    notes = sorted(transcripts_dir.rglob("*.md"), key=lambda p: str(p).lower())
    result = []

    for note in notes:
        if include and include.lower() not in str(note).lower():
            continue

        metadata, body, raw = read_note(note)
        status = str(metadata.get("summary_status", "pending")).lower()

        if force or status == "pending":
            result.append(note)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create AI summaries for transcript markdown notes"
    )

    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    obsidian_dir = Path(cfg["obsidian_dir"])
    transcripts_dir = obsidian_dir / "transcripts"

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts dir not found: {transcripts_dir}")

    notes = find_pending_notes(
        transcripts_dir,
        include=args.include,
        force=args.force,
    )

    if args.limit:
        notes = notes[:args.limit]

    print(f"Найдено заметок к summary: {len(notes)}")

    if args.list or args.dry_run:
        for note in notes:
            metadata, _, _ = read_note(note)
            print(f"[{metadata.get('summary_status', 'pending')}] {note}")
        return

    for note in tqdm(notes):
        try:
            metadata, body, raw = read_note(note)
            transcript = extract_section(body, "Расшифровка")

            if not transcript.strip() or "_Речь не распознана._" in transcript:
                mark_no_transcript(note, metadata, body)
                print(f"\nSKIP(no transcript): {note}")
                continue

            ai = call_ollama(cfg, metadata, transcript)
            new_path = update_note(note, ai, rename=args.rename)

            print(f"\nOK: {new_path}")

        except Exception as e:
            print(f"\nОшибка: {note}\n{e}\n")


if __name__ == "__main__":
    main()
