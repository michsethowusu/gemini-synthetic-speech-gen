"""Async synthesis runner: concurrency, rate limiting, retries, resume.

Resume is the design point. A long run *will* be interrupted — an API quota, a
dropped connection, a laptop lid. Each finished utterance writes its own audio
file plus a sidecar JSON, so restarting skips what exists and nothing is ever
appended to a shared file mid-flight. (Appending live to one metadata file is
what left the first Twi run with doubled and conflicting lines, forcing the
manifest to be rebuilt from disk afterwards.)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import tts as tts_registry
from .lang import Language
from .normalise import Normaliser
from .tts.base import RetryableTTSError, TTSError


@dataclass
class Utterance:
    index: int
    text: str
    transcript: str        # what the model is actually asked to speak
    # Per-utterance overrides. A single-language run leaves all three unset and
    # takes the run's single voice and language; the samples gallery sets them
    # so one pass can cover many languages, each with its own voice and its own
    # accent context, without a second copy of the retry/rate-limit machinery.
    voice: Optional[str] = None
    language: Optional[Language] = None
    name: Optional[str] = None

    @property
    def stem(self) -> str:
        return self.name or f"utt_{self.index:06d}"


class RateLimiter:
    """Token bucket over a rolling minute — keeps a run inside its RPM quota."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._times: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.rpm:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60.0]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                await asyncio.sleep(60.0 - (now - self._times[0]) + 0.01)


class Workspace:
    """On-disk layout of a run. Audio is bucketed so no directory gets huge."""

    def __init__(self, root: str):
        self.root = root
        self.audio_dir = os.path.join(root, "audio")
        self.meta_dir = os.path.join(root, "meta")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    def bucket(self, index: int) -> str:
        path = os.path.join(self.audio_dir, f"{index // 1000:03d}")
        os.makedirs(path, exist_ok=True)
        return path

    def audio_path(self, utterance: Utterance) -> str:
        return os.path.join(self.bucket(utterance.index), f"{utterance.stem}.wav")

    def meta_path(self, utterance: Utterance) -> str:
        return os.path.join(self.meta_dir, f"{utterance.stem}.json")

    def is_done(self, utterance: Utterance) -> bool:
        meta = self.meta_path(utterance)
        if not os.path.exists(meta):
            return False
        try:
            with open(meta, encoding="utf-8") as handle:
                record = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return False
        audio = record.get("audio_path")
        return bool(audio) and os.path.exists(audio) and os.path.getsize(audio) > 44

    def write(self, utterance: Utterance, record: dict) -> None:
        # Write the sidecar last: audio-then-metadata means a crash between the
        # two leaves the item simply unfinished, never falsely complete.
        tmp = self.meta_path(utterance) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        os.replace(tmp, self.meta_path(utterance))

    def records(self) -> List[dict]:
        out = []
        for name in sorted(os.listdir(self.meta_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.meta_dir, name), encoding="utf-8") as handle:
                try:
                    out.append(json.load(handle))
                except json.JSONDecodeError:
                    continue
        return sorted(out, key=lambda r: r.get("index", 0))


def build_utterances(sentences: Sequence[str], normaliser: Normaliser) -> List[Utterance]:
    utterances = []
    for index, text in enumerate(sentences):
        transcript = normaliser(text) if normaliser.enabled else text
        utterances.append(Utterance(index=index, text=text, transcript=transcript))
    return utterances


def render_prompt(config, language: Language, transcript: str) -> str:
    context = config.context.format(language=language.name, code=language.code)
    return config.prompt.format(context=context, text=transcript)


async def _one(utterance: Utterance, backend, workspace: Workspace, limiter: RateLimiter,
               semaphore: asyncio.Semaphore, config, language: Language,
               counters: dict) -> bool:
    async with semaphore:
        voice = utterance.voice or config.voice
        prompt = render_prompt(config, utterance.language or language, utterance.transcript)

        for attempt in range(1, config.max_retries + 1):
            await limiter.acquire()
            try:
                clip = await backend.synth(prompt, voice)
            except RetryableTTSError as exc:
                if attempt == config.max_retries:
                    counters["failed"] += 1
                    print(f"[{utterance.index}] giving up after {attempt} tries: {exc}", flush=True)
                    return False
                delay = min(60.0, 2 ** attempt) * (0.5 + random.random())
                print(f"[{utterance.index}] retry {attempt}/{config.max_retries} "
                      f"in {delay:.1f}s: {exc}", flush=True)
                await asyncio.sleep(delay)
                continue
            except TTSError as exc:
                counters["failed"] += 1
                print(f"[{utterance.index}] failed: {exc}", flush=True)
                return False

            path = workspace.audio_path(utterance)
            with open(path, "wb") as handle:
                handle.write(clip.audio)
            workspace.write(utterance, {
                "index": utterance.index,
                "name": utterance.stem,
                "language": (utterance.language or language).code,
                "text": utterance.text,
                "transcript": utterance.transcript,
                "voice": clip.voice,
                "audio_path": path,
                "sample_rate": clip.sample_rate,
                "bytes": len(clip.audio),
            })
            counters["done"] += 1
            total = counters["total"]
            if counters["done"] % 25 == 0 or counters["done"] == total:
                elapsed = time.time() - counters["started"]
                rate = counters["done"] / elapsed if elapsed else 0
                print(f"  {counters['done']}/{total} synthesised "
                      f"({rate * 60:.0f}/min, {counters['failed']} failed)", flush=True)
            return True
    return False


async def synthesise_async(utterances: Sequence[Utterance], config, language: Language,
                           work_dir: str, resume: bool = True) -> dict:
    workspace = Workspace(work_dir)
    pending = [u for u in utterances if not (resume and workspace.is_done(u))]
    skipped = len(utterances) - len(pending)
    if skipped:
        print(f"  resuming: {skipped} already done, {len(pending)} to go", flush=True)
    if not pending:
        return {"done": 0, "failed": 0, "skipped": skipped, "workspace": workspace}

    backend = tts_registry.get_backend(config.backend, config, language)
    num_keys = getattr(backend, "num_keys", 1)
    concurrency = config.concurrency
    rpm = config.rpm
    if num_keys > 1 and getattr(config, "scale_concurrency_with_keys", True):
        concurrency = config.concurrency * num_keys
        rpm = config.rpm * num_keys

    print(f"  backend: {backend.describe()}, voice={config.voice}, "
          f"concurrency={concurrency}, rpm={rpm}", flush=True)

    limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(concurrency)
    counters = {"done": 0, "failed": 0, "total": len(pending), "started": time.time()}
    try:
        await asyncio.gather(*[
            _one(u, backend, workspace, limiter, semaphore, config, language, counters)
            for u in pending
        ])
    finally:
        await backend.close()

    return {"done": counters["done"], "failed": counters["failed"],
            "skipped": skipped, "workspace": workspace}


def synthesise(utterances: Sequence[Utterance], config, language: Language,
               work_dir: str, resume: bool = True) -> dict:
    return asyncio.run(synthesise_async(utterances, config, language, work_dir, resume))
