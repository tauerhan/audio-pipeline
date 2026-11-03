import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_config import load_config

REQUIRED = (
    "README.md",
    "AGENTS.md",
    "PROGRESS.md",
    "feature_list.json",
    "config.example.yaml",
    "requirements.txt",
    "requirements-check.txt",
    "docs/ARCHITECTURE.md",
    "docs/AUDIT.md",
    "docs/DEVELOPMENT.md",
    "docs/HARNESS.md",
    "evals/README.md",
    ".github/workflows/check.yml",
)
PRIVATE_NAMES = {"config.yaml", "pipeline_state.json", "pipeline_index.csv", ".env"}
PRIVATE_DIRS = {"venv", ".venv", "data", "models", "transcripts", "processed_audio", ".obsidian"}
PRIVATE_SUFFIXES = {".key", ".pem", ".pub", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".mkv"}


def git(*arguments, input_text=None):
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def check_config(path):
    load_config(path, check_paths=False)


def check_repository():
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            raise ValueError(f"Missing repository file: {name}")
    python_files = list(ROOT.glob("*.py"))
    for directory in ("scripts", "src", "tests"):
        python_files.extend((ROOT / directory).rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Python syntax: {len(python_files)} files OK")
    check_config(ROOT / "config.example.yaml")
    for path in (ROOT / ".github").rglob("*.yml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))
    for path in (ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"):
        yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
    manifest = json.loads((ROOT / "evals/manifest.example.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("samples"), list):
        raise ValueError("Invalid evaluation example manifest")
    features = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))
    if not isinstance(features, list) or not features:
        raise ValueError("Feature list must be nonempty")
    identifiers = set()
    for feature in features:
        for name in ("id", "title", "behavior", "verification", "state"):
            if not isinstance(feature.get(name), str) or not feature[name].strip():
                raise ValueError(f"Feature missing {name}")
        if feature["id"] in identifiers:
            raise ValueError("Duplicate feature id")
        identifiers.add(feature["id"])
        if feature["state"] not in {"not_started", "active", "blocked", "passing"}:
            raise ValueError("Invalid feature state")
        if feature["state"] == "passing" and not feature.get("evidence"):
            raise ValueError("Passing feature requires verification evidence")
    print("Example config, workflow, evaluation manifest and feature list: OK")
    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != ROOT:
        raise ValueError("Initialize Git in the audio-pipeline directory itself")
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    for name in filter(None, paths):
        path = Path(name)
        if path.name in PRIVATE_NAMES or PRIVATE_DIRS.intersection(path.parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
            raise ValueError(f"Private data would enter Git: {name}")
        if path.name.startswith((".env.", "id_rsa", "id_ed25519")) and path.name != ".env.example":
            raise ValueError(f"Private configuration or key would enter Git: {name}")
    probes = [
        "config.yaml", "venv/bin/python", ".venv/bin/python", "data/inbox/sample.mp3",
        "models/weights.bin", "transcripts/private.md", "processed_audio/sample.wav",
        "pipeline_state.json", "pipeline_index.csv", "connection.key", ".env",
        "evals/manifest.local.json", "evals/results/measurement.json",
    ]
    ignored = set(git("check-ignore", "--no-index", "--stdin", input_text="\n".join(probes) + "\n").splitlines())
    if set(probes) - ignored:
        raise ValueError("Missing private-data ignore rules: " + ", ".join(sorted(set(probes) - ignored)))
    print("Git private-data exclusions: OK")
    print("Repository check passed. ASR quality and end-to-end behavior were NOT tested.")


def main():
    parser = argparse.ArgumentParser(description="Read-only repository checks without models or audio processing")
    parser.add_argument("--config", type=Path, help="Only validate a local config; do not run the pipeline")
    arguments = parser.parse_args()
    try:
        if arguments.config:
            check_config(arguments.config)
            print("Local config structure: OK. Paths, models and services were not tested.")
        else:
            check_repository()
    except yaml.YAMLError as error:
        location = getattr(error, "problem_mark", None)
        position = f" at line {location.line + 1}, column {location.column + 1}" if location else ""
        print(f"CHECK FAILED: invalid YAML{position}; configuration values omitted", file=sys.stderr)
        return 1
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
