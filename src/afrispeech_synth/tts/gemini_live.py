"""Gemini Live API backend — conversational audio models used as a TTS engine.

Why this exists alongside `gemini.py`: the Live models are on a different quota
from the TTS models, and the native-audio ones read several African languages
more faithfully than the TTS line does. The trade-off is that they are
*conversational* — left alone they will answer the sentence rather than read
it — so this backend leans on a system instruction to pin them to reading.

Two things shape the implementation:

* **A session is a websocket, not a request.** So the backend keeps a pool of
  open sessions and hands one to each concurrent utterance, instead of paying a
  connect on every clip.
* **A session's voice is fixed when it connects.** The runner rotates voices
  per utterance, so the pool is keyed by voice — asking for Puck never gets
  handed back a Zephyr session.

Sessions are also recycled every `session_turns` utterances: the Live API keeps
the whole turn history in context, so a long-lived session slows down and
starts letting earlier sentences bleed into later reads.
"""
from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple

from .base import (Clip, RetryableTTSError, TTSBackend, TTSError, pcm_to_wav,
                   wav_sample_rate)

# The Live API returns headerless little-endian PCM, declared as
# `audio/pcm;rate=24000` -- note `audio/pcm`, not the `audio/L16` the TTS
# models send, so the bit depth is not in the mime type and 16 is assumed.
# Measured: real spectral energy above 8 kHz with no cliff there, so this is a
# genuine 24 kHz signal rather than something upsampled from 16 kHz.
# The declared rate is still read per response rather than trusted from here.
LIVE_RATE = 24000

# Websocket close codes the Live API uses for conditions that pass: 1007 and
# 1011 are server-side faults, 1008 is the session being abandoned mid-turn
# ("the operation was aborted") under concurrency, and 1012/1013 are restarts.
# A dropped session is not a bad sentence — the same text succeeds on a fresh
# socket, so anything that only says the connection went away must be retried.
# Treating 1008 as fatal silently dropped 2.4% of a full run.
RETRYABLE_MARKERS = ("429", "rate limit", "resource_exhausted", "quota",
                     "500", "502", "503", "504", "unavailable", "deadline",
                     "timeout", "internal error", "connection closed",
                     "going away", "keepalive", "aborted", "reset",
                     "1007", "1008", "1011", "1012", "1013")


class _Session:
    """One open Live websocket plus the turn count that decides its retirement."""

    __slots__ = ("ctx", "session", "voice", "turns")

    def __init__(self, ctx, session, voice):
        self.ctx = ctx
        self.session = session
        self.voice = voice
        self.turns = 0

    async def close(self) -> None:
        try:
            await self.ctx.__aexit__(None, None, None)
        except Exception:
            pass        # a socket that is already gone is still closed


class GeminiLiveTTS(TTSBackend):
    name = "gemini-live"

    def __init__(self, config, language):
        super().__init__(config, language)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:      # pragma: no cover
            raise TTSError(
                "The Gemini Live backend needs the google-genai SDK:\n"
                "    pip install 'afrispeech-synth[gemini]'"
            ) from exc
        self._types = types

        from ..env import load as load_env

        keys_env = getattr(config, "api_keys_env", "GEMINI_API_KEYS")
        key_env = getattr(config, "api_key_env", "GEMINI_API_KEY")
        raw_keys = os.environ.get(keys_env) or os.environ.get(key_env)
        if not raw_keys:
            load_env()
            raw_keys = os.environ.get(keys_env) or os.environ.get(key_env)
        if not raw_keys:
            raise TTSError(
                f"No API key found. Either:\n"
                f"    export {key_env}=...\n"
                f"or export {keys_env}=key1,key2,key3 (for multi-key pooling)\n"
                f"or put `{key_env}=...` in a .env file in this directory.\n"
                f"Keys must never go in a config file — those get committed."
            )
        self._keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self._clients = [genai.Client(api_key=k) for k in self._keys]
        self._client = self._clients[0]
        self._client_idx = 0
        self.num_keys = len(self._clients)

        from ..live_models import DEFAULT, MODELS, names, unknown
        # `tts.model` defaults to a TTS model, which the Live API cannot serve.
        # Rather than fail on a config that only named the backend, fall back to
        # the Live default and say so.
        if "tts" in config.model and unknown(config.model):
            print(f"  note: {config.model} is a TTS model, not a Live one; "
                  f"using {DEFAULT}", flush=True)
            config = config.__class__(**{**vars(config), "model": DEFAULT})
            self.config = config
        if unknown(config.model):
            print(f"  note: {config.model} is not one of the models this build has "
                  f"been probed on ({', '.join(names())}). It may work; it has not "
                  f"been checked on African languages.", flush=True)
        self._model_label = MODELS[config.model].label if config.model in MODELS else config.model
        self._idle: Dict[str, List[_Session]] = {}
        self._open: List[_Session] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ config

    def _connect_config(self, voice: str):
        types = self._types
        kwargs = dict(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )
        instruction = getattr(self.config, "system_instruction", None)
        if instruction:
            kwargs["system_instruction"] = types.Content(
                parts=[types.Part(text=instruction)])
        # gemini-3.8-live-extended-thinking refuses to connect without a level.
        level = getattr(self.config, "thinking_level", None)
        if level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        return types.LiveConnectConfig(**kwargs)

    # ------------------------------------------------------------------- pool

    async def _acquire(self, voice: str) -> _Session:
        async with self._lock:
            pool = self._idle.get(voice)
            if pool:
                return pool.pop()
            client = self._clients[self._client_idx % len(self._clients)]
            self._client_idx += 1
        try:
            ctx = client.aio.live.connect(
                model=self.config.model, config=self._connect_config(voice))
            entry = _Session(ctx, await ctx.__aenter__(), voice)
        except Exception as exc:
            raise self._wrap(exc) from exc
        async with self._lock:
            self._open.append(entry)
        return entry

    async def _release(self, entry: _Session, reusable: bool) -> None:
        limit = getattr(self.config, "session_turns", 25)
        if not reusable or (limit and entry.turns >= limit):
            async with self._lock:
                if entry in self._open:
                    self._open.remove(entry)
            await entry.close()
            return
        async with self._lock:
            self._idle.setdefault(entry.voice, []).append(entry)

    @staticmethod
    def _wrap(exc: Exception) -> TTSError:
        message = str(exc).lower()
        if any(marker in message for marker in RETRYABLE_MARKERS):
            return RetryableTTSError(str(exc))
        return TTSError(str(exc))

    # ------------------------------------------------------------------ synth

    async def _turn(self, session, text: str) -> Tuple[bytes, str, str]:
        types = self._types
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True)
        pcm, spoken, mime = bytearray(), [], ""
        async for message in session.receive():
            content = message.server_content
            if not content:
                continue
            if content.model_turn:
                for part in content.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        pcm.extend(part.inline_data.data)
                        mime = part.inline_data.mime_type or mime
                    elif part.text and not getattr(part, "thought", False):
                        spoken.append(part.text)
            if content.output_transcription and content.output_transcription.text:
                spoken.append(content.output_transcription.text)
            if content.turn_complete:
                break
        return bytes(pcm), "".join(spoken).strip(), mime

    async def synth(self, text: str, voice: str) -> Clip:
        entry = await self._acquire(voice)
        try:
            pcm, _, mime = await self._turn(entry.session, text)
            entry.turns += 1
        except Exception as exc:
            # A broken turn can leave the socket unusable, so the session is
            # retired rather than handed to the next utterance.
            await self._release(entry, reusable=False)
            raise self._wrap(exc) from exc

        if not pcm:
            # Usually a safety block or a silently dropped turn; retrying works
            # often enough that it is treated as transient, as in the TTS backend.
            await self._release(entry, reusable=False)
            raise RetryableTTSError("model returned no audio")

        await self._release(entry, reusable=True)
        wav = pcm_to_wav(pcm, mime, default_rate=LIVE_RATE)
        # The rate the response actually declared, not the one assumed above.
        rate = wav_sample_rate(wav) or LIVE_RATE
        return Clip(audio=wav, mime_type="audio/wav", sample_rate=rate, voice=voice)

    def describe(self) -> str:
        key_info = f", {self.num_keys} API keys in pool" if self.num_keys > 1 else ""
        return f"{self.name} ({self._model_label}{key_info})"

    async def close(self) -> None:
        async with self._lock:
            entries, self._open, self._idle = list(self._open), [], {}
        for entry in entries:
            await entry.close()
