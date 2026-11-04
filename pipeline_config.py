from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


def validate_config(config):
    if not isinstance(config, dict):
        raise ConfigError("Конфигурация должна быть YAML-словарём")
    for name in ("source_dir", "obsidian_dir", "model", "language"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ConfigError(f"Нужно непустое строковое поле {name}")
        if "\0" in config[name]:
            raise ConfigError(f"Поле {name} содержит недопустимый нулевой символ")
    if config["language"] != "ru":
        raise ConfigError("Для этого проекта language должен быть ru")
    extensions = config.get("audio_extensions")
    if not isinstance(extensions, list) or not extensions:
        raise ConfigError("audio_extensions должен быть непустым списком")
    if not all(isinstance(value, str) and len(value) > 1 and value.startswith(".") for value in extensions):
        raise ConfigError("Каждое расширение audio_extensions должно начинаться с точки")
    for section in ("convert", "transcription"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Нужен словарь {section}")
    for name in ("sample_rate", "channels"):
        value = config["convert"].get(name)
        if type(value) is not int or value <= 0:
            raise ConfigError(f"convert.{name} должен быть положительным целым числом")
    audio_filter = config["convert"].get("audio_filter")
    if audio_filter is not None and (not isinstance(audio_filter, str) or not audio_filter.strip()):
        raise ConfigError("convert.audio_filter должен быть непустой строкой или null")
    for name in ("device", "compute_type"):
        value = config["transcription"].get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Нужно непустое строковое поле transcription.{name}")
    if config["transcription"]["device"] not in {"cpu", "cuda", "auto"}:
        raise ConfigError("transcription.device должен быть cpu, cuda или auto")
    compute_types = {
        "default", "auto", "float32", "float16", "bfloat16", "int16",
        "int8", "int8_float32", "int8_float16", "int8_bfloat16",
    }
    if config["transcription"]["compute_type"] not in compute_types:
        raise ConfigError("Неизвестный transcription.compute_type")
    if type(config["transcription"].get("vad_filter")) is not bool:
        raise ConfigError("transcription.vad_filter должен быть true или false")
    if type(config["transcription"].get("condition_on_previous_text", True)) is not bool:
        raise ConfigError("transcription.condition_on_previous_text должен быть true или false")
    initial_prompt = config["transcription"].get("initial_prompt")
    if initial_prompt is not None and not isinstance(initial_prompt, str):
        raise ConfigError("transcription.initial_prompt должен быть строкой или null")
    prompt_reset = config["transcription"].get("prompt_reset_on_temperature", 0.5)
    if isinstance(prompt_reset, bool) or not isinstance(prompt_reset, (int, float)) or not 0 <= prompt_reset <= 1:
        raise ConfigError("transcription.prompt_reset_on_temperature должен быть числом от 0 до 1")
    hotwords = config["transcription"].get("hotwords", [])
    if not isinstance(hotwords, (str, list)) or (isinstance(hotwords, list) and not all(isinstance(item, str) for item in hotwords)):
        raise ConfigError("transcription.hotwords должен быть строкой или списком строк")
    ai = config.get("ai", {})
    if not isinstance(ai, dict):
        raise ConfigError("ai должен быть словарём")
    if type(ai.get("enabled", False)) is not bool:
        raise ConfigError("ai.enabled должен быть true или false")
    for name in ("url", "ollama_url", "model"):
        if name in ai and (not isinstance(ai[name], str) or not ai[name].strip()):
            raise ConfigError(f"ai.{name} должен быть непустой строкой")


def load_config(config_path="config.yaml", *, check_paths=True):
    path = Path(config_path).expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f", строка {mark.line + 1}, столбец {mark.column + 1}" if mark else ""
        raise ConfigError(f"Некорректный YAML{location}") from error
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"Не удалось прочитать конфигурацию: {path}") from error
    validate_config(config)
    for name in ("source_dir", "obsidian_dir"):
        directory = Path(config[name]).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        config[name] = str(directory.resolve())
    if check_paths:
        source = Path(config["source_dir"])
        if not source.is_dir():
            raise ConfigError(f"Папка source_dir не существует или не является каталогом: {source}")
        output = Path(config["obsidian_dir"])
        for directory in (output, output / "transcripts", output / "processed_audio"):
            for candidate in (directory, *directory.parents):
                if candidate.exists() and not candidate.is_dir():
                    raise ConfigError(f"Путь вывода занят файлом: {candidate}")
    return config
