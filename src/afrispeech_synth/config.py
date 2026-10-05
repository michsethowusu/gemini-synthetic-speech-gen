"""Run configuration: one YAML file describes a whole dataset build.

Every stage reads from this object, so a run is reproducible from the file
alone. CLI flags override individual fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, List, Optional

import yaml

DEFAULT_PROMPT = """## Sample Context:
{context}

## Transcript:
{text}"""


LIVE_SYSTEM_INSTRUCTION = (
    "You are a text-to-speech engine. The user message contains a transcript, "
    "possibly preceded by context lines describing how to speak it. Read the "
    "transcript aloud verbatim, in its own language, applying that context. "
    "Never translate it, never answer it, never comment on it, and never add "
    "or omit words. Speak only the transcript and nothing else."
)


@dataclass
class SelectConfig:
    # "none" keeps every sentence; "word" and "phoneme" run greedy set cover.
    cover: str = "phoneme"
    min_freq: int = 1
    max_sentences: Optional[int] = None
    min_chars: int = 8
    max_chars: int = 240
    seed: int = 0


@dataclass
class TTSConfig:
    backend: str = "gemini"
    model: str = "gemini-3.1-flash-tts-preview"
    # One voice per dataset. A dataset is normally one speaker, so this is a
    # single name, not a rotation — `afrispeech-synth voices` lists the choices
    # and the samples gallery lets you hear them before picking.
    voice: str = "Zephyr"
    # Gallery only: the pool the samples command spreads across languages so a
    # demo page covers the catalogue. It has no effect on a dataset run.
    voices: List[str] = field(default_factory=list)
    # {context} and {text} are filled in per utterance.
    prompt: str = DEFAULT_PROMPT
    context: str = "speak in {language} accent"
    temperature: float = 1.0
    concurrency: int = 10
    rpm: int = 200
    max_retries: int = 5
    sample_rate: int = 24000
    api_key_env: str = "GEMINI_API_KEY"
    api_keys_env: str = "GEMINI_API_KEYS"
    scale_concurrency_with_keys: bool = True

    # --- gemini-live backend only -----------------------------------------
    # The Live models are conversational: without an instruction pinning them
    # to reading, they answer the transcript instead of speaking it.
    system_instruction: str = LIVE_SYSTEM_INSTRUCTION
    # Retire a Live session after this many utterances. The API keeps the whole
    # turn history in context, so a session left open indefinitely slows down
    # and starts letting earlier sentences bleed into later reads. 0 disables.
    session_turns: int = 25
    # Required by gemini-3.8-live-extended-thinking, ignored by other models.
    thinking_level: Optional[str] = None


@dataclass
class PackageConfig:
    shard_target_mb: int = 190
    formats: List[str] = field(default_factory=lambda: ["parquet"])
    push_to: Optional[str] = None   # e.g. "AfriSpeech/twi-synthetic-speech"
    private: bool = False


@dataclass
class RunConfig:
    language: str = "twi"
    sources: List[str] = field(default_factory=list)
    # universal | grapheme | ipa | none — what goes to the TTS model as the
    # transcript. Universal is the default: it maps every phoneme onto the letter
    # most African languages use for it, which TTS voices read more reliably than
    # language-specific characters like ɔ, ɛ or stacked tone diacritics.
    normalise: str = "universal"
    out: str = "out"
    work: Optional[str] = None      # defaults to <out>/work
    select: SelectConfig = field(default_factory=SelectConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    package: PackageConfig = field(default_factory=PackageConfig)

    @property
    def work_dir(self) -> str:
        return self.work or os.path.join(self.out, "work")

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {"select": SelectConfig, "tts": TTSConfig, "package": PackageConfig}


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    return cls(**data)


def from_dict(data: dict) -> RunConfig:
    data = dict(data or {})
    sections = {name: _build(cls, data.pop(name, {}) or {}) for name, cls in _SECTIONS.items()}
    if isinstance(data.get("sources"), str):
        data["sources"] = [data["sources"]]
    return _build(RunConfig, {**data, **sections})


def load(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as handle:
        return from_dict(yaml.safe_load(handle))


def apply_overrides(config: RunConfig, overrides: dict) -> RunConfig:
    """Apply CLI overrides. Dotted keys address a section: `tts.voices`."""
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            section, attr = key.split(".", 1)
            setattr(getattr(config, section), attr, value)
        else:
            setattr(config, key, value)
    return config


def dump(config: RunConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
