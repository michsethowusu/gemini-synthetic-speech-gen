# gemini-synthetic-speech-gen

**Turn [Google Gemini](https://ai.google.dev/gemini-api/docs) into a speech-dataset factory
for any African language — from raw text to a training-ready HuggingFace dataset, with one
command.**

*(Formerly `afrispeech-synth`, enhanced with multi-key pool scaling).*

Most African languages have no recorded speech corpus. Gemini can speak many of them well
enough to bootstrap one, but a usable dataset is not just API calls: you need text in the
language, the *right* sentences rather than all of them, orthography the model reads
correctly, and output packaged so a trainer can consume it. This does all four around Gemini,
through either the **TTS** models or the **Live** API's native-audio models.

```bash
pip install "gemini-synthetic-speech-gen[gemini]"
export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey

# Or scale 5x faster by pooling multiple API keys:
export GEMINI_API_KEYS="key1,key2,key3,key4,key5"

gemini-synthetic-speech-gen run examples/twi.yaml
```

**[Hear what it produces](https://huggingface.co/spaces/AfriSpeech/afrispeech-synth-demo)** —
one sample per language, 562 languages, a different voice each.

## What's around Gemini

| | |
|---|---|
| **The voice** | **Google Gemini**, [30 voices](#4--synthesise), your own API key. Two backends: [TTS](https://ai.google.dev/gemini-api/docs/speech-generation) (`gemini`, the reference — every dataset built with this tool so far was spoken by it) and the [Live API](https://ai.google.dev/gemini-api/docs/live) (`gemini-live`, a separate quota, better on several African languages). |
| **The text** | [africa-corpus-builder](https://github.com/AfriSpeech/africa-corpus-builder) — source text for **693 African languages**, so a language with no corpus of its own still has a starting point |
| **The orthography** | [africa-g2p](https://github.com/AfriSpeech/africa-g2p) — phoneme tables for **400 languages** (**408** once aliases resolve). Feeding Gemini a language's raw orthography gets you its guess at `ɔ`, `ɛ` and `ŋ`; feeding it the universal grapheme set gets you the sound |
| **The names** | [afriso](https://github.com/AfriSpeech/afriso) — resolves `Twi`, `tw`, `aka`, `Asante Twi` to one code all of the above agree on |
| **The selection** | Greedy set cover over phoneme units — on Twi, 4,141 candidate sentences reduced to 115 at full phoneme coverage. 97% fewer Gemini calls for the same coverage |

Gemini is the reference backend, not a dependency of the design — [swapping it
out](#adding-a-tts-backend) is one method and one line of config — but it is what this was
built on and tuned against, and the parts above exist because raw API calls alone did not
produce a dataset worth training on.

For **recorded** African speech rather than synthetic, use
[afrispeech-selector](https://github.com/AfriSpeech/afrispeech-selector).

---

## Which languages work

| | Languages | What you need to do |
|---|--:|---|
| **Ready** | **562** | Nothing. Name the language and run. |
| **Bring your own text** | 185 | Point a `file:` or `hf:` source at your own sentences. |
| **No G2P table** | 131 | Text is available; run with `--normalise none`. |

africa-g2p converts **747** African languages to universal orthography,
africa-corpus-builder has text for **693**, and **562 are in both** — those need nothing
from you but a name:

These counts are measured, not read off a list. africa-g2p exposes two different things —
`registry()`, its metadata table, and `available_languages()`, what it can actually convert —
and they are not the same size, because rule tables get added faster than metadata rows. Asking
the registry reported every language with a table but no description of itself as unsupported,
and the same key matching dropped eight more (Akan, Luo, Luwo, Mwan, Ngemba, Kamba, Malgache,
Tonga) that reach a table through an alias. africa-g2p also marks some languages as ones it
cannot yet write in universal; those are excluded rather than counted and left to fail at run
time. Readiness now asks whether the language really converts, so the number tracks africa-g2p as tables land instead of lagging a release behind.

```bash
afrispeech-synth run --lang Zulu --source corpus:zul --out out/zul
```

**If your language has no corpus text, it still works — supply your own:**

```bash
afrispeech-synth run --lang Afar --source file:my_afar_sentences.txt --out out/aar
afrispeech-synth run --lang Afar --source hf:my-org/my-dataset#text  --out out/aar
```

Any source works for any language, and you can mix them. If there's no G2P table either, add
`--normalise none` and the TTS model gets your raw text — everything else in the pipeline is
unchanged.

Check where yours stands:

```bash
afrispeech-synth langs --search zulu     # zul  Zulu  ready  Atlantic-Congo
afrispeech-synth langs --ready           # the 562 that need nothing from you
```

Languages are matched by exact code only. Matching by name would add ~73 more, but it pairs
different languages that share an alternative name — Tunisian Arabic text under an Algerian
Arabic table, Basa of Cameroon under Basa of Nigeria — so those are left out rather than
shipped wrong.

---

## The pipeline

```
 source ──► select ──► normalise ──► synthesise ──► package
  text      phoneme      africa-g2p      TTS API     parquet + card
   │         cover          │              │            │
   └ corpus:twi             └ grapheme     └ resumable  └ push to the Hub
     hf:org/ds#col            or IPA         + retries
     file:x.txt
```

**1 · Source.** Combine any number of text sources; duplicates are dropped in first-seen order.

```yaml
sources:
  - corpus:twi                                               # africa-corpus-builder
  - hf:ghanaopenai/Ghana_English-Twi_Code-switching_Speech#transcript
  - file:my_sentences.txt
```

**2 · Select.** Greedy set cover over **phoneme** units (default) or **word** units: the fewest
sentences that still contain every sound. Every sentence dropped is a TTS call you don't pay for —
on Twi it cut a 4,141-sentence pool to 115 at full phoneme coverage.

```
  covered 208/208 phoneme units (100.0%) with 115 sentences
```

**3 · Normalise.** `africa-g2p` rewrites each sentence, then punctuation is reduced to `.` `?`
`!` `,` — the marks a voice uses for phrasing. Everything else (apostrophes, asterisks marking
proper nouns, hyphens, colons, quotes) is read as a pause or spelled out, so it goes. Stored as
`normalised_text` — it's what the TTS model is actually asked to speak.

| `--normalise` | Twi example | When |
|---|---|---|
| `universal` *(default)* | `ho bobea onyankopon` | Every phoneme written with the letter most African languages use for it (`ɔ`→`o`, `ɛ`→`e`). Plain `a-z` only — needs africa-g2p ≥ 0.2.3. |
| `grapheme` | `hɔ bɔbea onyankopɔn` | The language's own phoneme units, multigraphs (`ny`, `kp`) kept whole and special characters preserved. |
| `ipa` | `hɔ bɔbea oɲankʰopʰɔn` | Phonetic symbols — for phoneme-level ASR work, not for speech generation. |
| `none` | `hɔ bɔbea Onyankopɔn` | Send the raw text. Works for any language, G2P table or not. |

**Whichever you pick, look at the output before a full run.** Neither transform is right for every
language:

```
           universal                    grapheme
twi        na ho bobea            ✓     na hɔ bɔbea           ɔ/ɛ may be mispronounced
yor        ngi˥ i˩bɛ˩rɛ˩          ✗     ní ìbẹ̀rẹ̀              ✓
afr        eng die aarde khemaak  ✗     en die aarde gemaak   ✓
```

Universal is what the Ghana Twi dataset was built with and is the default, but it rewrites more
than it should in some languages. `--dry-run` prints the selection without spending any API
calls; `afrispeech-synth card config.yaml` shows the transform that will be recorded:

```bash
afrispeech-synth run --lang yor --source corpus:yor --normalise grapheme --dry-run
```

**4 · Synthesise.** Google Gemini TTS speaks the normalised transcript, in any of its
**30 voices**:

```bash
afrispeech-synth voices            # Zephyr Bright, Kore Firm, Sulafat Warm, …
--voice Zephyr                        # the dataset's speaker (default)
--voice Sulafat                       # pick another after hearing it in the gallery
```

A dataset is **one voice**. `afrispeech-synth voices` lists all 30, and the
[samples gallery](#samples-gallery) lets you hear each one before you commit a run to it.

**Two Gemini backends.** `gemini` is the TTS line; `gemini-live` drives the Live API's
conversational audio models, which sit on a **separate quota** and read several African
languages more faithfully:

```yaml
tts:
  backend: gemini-live
  model: models/gemini-2.5-flash-native-audio-latest
  voice: Zephyr
```

The Live models are conversational, so left alone they *answer* the transcript instead of
reading it. The backend pins them to reading with a system instruction (`tts.system_instruction`),
holds a pool of websocket sessions rather than reconnecting per clip, and retires each session
every `tts.session_turns` utterances so earlier sentences do not bleed into later reads.

Three Live models generate speech, and `afrispeech-synth models` lists them with what a
probe actually found — one sentence in each of Twi, Ewe, Dagbani, Ga and Hausa, scored
against what the model said it spoke:

| `tts.model` | Probe | Notes |
|---|---|---|
| `models/gemini-2.5-flash-native-audio-latest` | 5/5 clean | Read every sentence back verbatim. Brighter — energy reaches ~9.7 kHz against ~5.6 kHz. |
| `models/gemini-3.1-flash-live-preview` **(default)** | 5/5 clean | Verbatim, and a little faster. What the sample gallery is built with. |
| `models/gemini-3.8-live` | 2/5 clean | Newest but weakest here: no transcript for Twi or Ewe, a 0.8s truncated clip for Dagbani. Probe your language first. |

`gemini-3.8-live-extended-thinking` additionally requires `tts.thinking_level`. The Live
API's other models are not speech generators and are deliberately absent:
`gemini-3.5-transcribe-live` is ASR, and `gemini-3.5-live-translate-preview` translates
rather than reads — the exact behaviour the system instruction exists to prevent.

A model outside this list still runs; the backend just notes that it has not been checked
on African languages.

**Multi-Key Pool Scaling (5x+ Speed).** Free-tier Gemini accounts enforce strict concurrent
session limits (typically 5 concurrent sessions for the Live API). To scale throughput
linearly without hitting 429 or 1011 quota errors, pass multiple Gemini API keys:

```bash
export GEMINI_API_KEYS="key1,key2,key3,key4,key5"
```

Or declare `GEMINI_API_KEYS=key1,key2,...` in a local `.env` file. The engine pools the keys,
rotates WebSocket connections and HTTP streams across them, and scales both concurrency and
RPM proportionally (e.g. 5 keys = 20 concurrent sessions, achieving ~180–200 clips/minute).

**What the audio is.** Clips are written exactly as the API returns them — this tool does
no resampling, no re-encoding and no format conversion. Both Gemini backends return the
same thing:

| | |
|---|---|
| Container | WAV (a RIFF header is added; the API sends headerless PCM) |
| Encoding | 16-bit signed little-endian PCM |
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |

The Live API declares `audio/pcm;rate=24000`, the TTS models `audio/L16;rate=24000`. That
24 kHz is real rather than upsampled: the signal carries measurable energy above 8 kHz with
no cliff there, which is what a 16 kHz source stretched to 24 would show.

If your trainer wants 16 kHz or a compressed format, resample downstream — `ffmpeg`, `sox`
or `torchaudio` do it better than a wrapper here would, and keeping the original means the
dataset never bakes in a lossy step you cannot undo.

Async, rate-limited, and **resumable**: every finished clip writes its own
audio file plus a sidecar record, so an interrupted run restarts where it stopped. Retries back
off on 429s and empty responses.

**5 · Package.** Parquet shards with audio bytes embedded (the viewer plays them inline), a
`metadata.jsonl` manifest, an LJSpeech export for Piper/VITS/MeloTTS, and a dataset card
generated from the run config.

---

## Install

```bash
pip install afrispeech-synth            # core
pip install "afrispeech-synth[gemini]"  # + the Gemini TTS backend
pip install "afrispeech-synth[hf]"      # + hf: text sources (pulls datasets)
```

`afriso` and `africa-corpus-builder` are not on PyPI yet:

```bash
pip install "afriso @ git+https://github.com/AfriSpeech/afriso"

git clone https://github.com/AfriSpeech/africa-corpus-builder
export AFRICA_CORPUS_PATH=$PWD/africa-corpus-builder
```

Both are optional. Without `afriso` you pass codes rather than names; without
africa-corpus-builder every source except `corpus:` still works.

---

## Usage

### Start from nothing

```bash
afrispeech-synth init Yoruba          # writes yor.yaml
afrispeech-synth run yor.yaml --dry-run   # select sentences, call no API
afrispeech-synth run yor.yaml
```

### Or stay on the command line

```bash
afrispeech-synth run \
  --lang Twi \
  --source corpus:twi \
  --cover phoneme \
  --max-sentences 2000 \
  --voice Zephyr \
  --out out/twi \
  --repo AfriSpeech/twi-synthetic-speech
```

### Samples gallery

**[Hear it: AfriSpeech/afrispeech-synth-demo](https://huggingface.co/spaces/AfriSpeech/afrispeech-synth-demo)**
— every language, in every voice. The page is built from this repo, so it has no repo of
its own:

```bash
afrispeech-synth samples --limit 20          # generate clips into space/
afrispeech-synth space                       # build space/index.html, preview locally
afrispeech-synth space --repo org/my-samples # publish it

# Full voice matrix: clips in a dataset, the Space just streams them
afrispeech-synth samples --all-voices
afrispeech-synth space \
  --repo AfriSpeech/afrispeech-synth-demo \
  --audio-repo AfriSpeech/synthetic-voice-samples-africa
```

`--distinct` gives each voice its own sentence instead of repeating one. That turns the
gallery into a small speech dataset — 30 distinct sentences per language, one per voice —
at the cost of comparability, since the voices no longer read the same line.

`--audio-repo` puts the clips in a **dataset** repo and leaves the Space as just the page.
Once the gallery carries every voice for every language that is thousands of files and
hundreds of megabytes — which belongs somewhere people can load and cite, not buried in a
Space. The page then streams from the dataset.

`samples` covers every ready language by default and spreads the 30 voices evenly across
them, so the gallery is also the voice catalogue. `--all-voices` instead reads every language
in *every* voice — the same sentence throughout, which is what makes voices comparable, since
switching voice on a card changes the voice and nothing else. That is 30x the clips for the
same languages (~6,500 for the full set, a few hundred MB), so it is worth a `--limit` run
first. Clips are published as generated — 24 kHz mono WAV. `--compress` re-encodes them to 64k MP3, which is only worth it when the audio ships inside the Space rather than in a dataset.

```bash
afrispeech-synth samples --all-voices --limit 3   # 3 languages x 30 voices, to preview
afrispeech-synth samples --all-voices             # the full matrix
afrispeech-synth samples --all-voices --distinct  # a different sentence per voice
```

### One stage at a time

```bash
afrispeech-synth select  config.yaml   # source + cover, writes sentences.txt
afrispeech-synth synth   config.yaml   # synthesise (resumes by default)
afrispeech-synth package config.yaml   # parquet + manifest + card
afrispeech-synth push    config.yaml --repo org/name

afrispeech-synth langs --ready         # languages that need no text from you
afrispeech-synth langs --search yor    # what one language needs
afrispeech-synth voices                # the 30 voices you can pick from
```

Interrupted? Run the same command again — finished clips are skipped.

### As a library

```python
from afrispeech_synth import RunConfig, run

config = RunConfig(language="Twi", sources=["corpus:twi"], out="out/twi")
config.select.cover = "phoneme"
config.select.max_sentences = 2000
config.tts.voice = "Zephyr"

run(config)
```

Individual stages are importable too:

```python
from afrispeech_synth import resolve, Normaliser, stage_sources, stage_select

language   = resolve("Twi")
normaliser = Normaliser(language, "grapheme")
sentences  = stage_sources(config, language)
selection  = stage_select(config, language, sentences, normaliser)

print(selection.coverage, len(selection.sentences))
```

---

## Configuration

```yaml
language: Twi                 # name, ISO 639-1/2/3 code, or alternative name
sources:                      # corpus: | hf: | file:
  - corpus:twi
normalise: universal          # universal | grapheme | ipa | none
out: out/twi

select:
  cover: phoneme              # phoneme | word | none
  min_freq: 1                 # only target units seen at least this often
  max_sentences: 12000
  min_chars: 20
  max_chars: 240
  seed: 0

tts:
  backend: gemini             # or gemini-live (see below)
  model: gemini-3.1-flash-tts-preview
  voice: Zephyr               # one speaker per dataset
  context: speak in {language} accent
  concurrency: 10
  rpm: 200                    # requests per minute, enforced
  max_retries: 5
  sample_rate: 24000
  api_key_env: GEMINI_API_KEY # the key is read from the environment, never the file

package:
  formats: [parquet, ljspeech]
  shard_target_mb: 190
  push_to: AfriSpeech/twi-synthetic-speech
  private: false
```

CLI flags override any field. API keys are read only from the environment — never
put one in a config file you intend to commit.

---

## Output

```
out/twi/
├── data/train-00000-of-00019.parquet   # audio bytes embedded, viewer-playable
├── metadata.jsonl                      # index, text, normalised_text, voice, shard
├── sentences.txt                       # the selected source sentences
├── README.md                           # generated dataset card
└── work/                               # per-clip audio + sidecars (resume state)
```

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="out/twi/data/*.parquet", split="train")
ds[0]["audio"]["array"], ds[0]["text"], ds[0]["normalised_text"]
```

---

## Adding a TTS backend

A backend is one method. Register it and it becomes available as `tts.backend`:

```python
from afrispeech_synth import tts
from afrispeech_synth.tts.base import Clip, TTSBackend

class MyTTS(TTSBackend):
    name = "mytts"
    async def synth(self, text: str, voice: str) -> Clip:
        audio = await my_api(text, voice)          # bytes
        return Clip(audio=audio, mime_type="audio/wav",
                    sample_rate=self.config.sample_rate, voice=voice)

tts.register("mytts", lambda: MyTTS)
```

Raise `RetryableTTSError` for rate limits and transient failures; the runner backs off
and retries. Raise `TTSError` for anything permanent.

---

## A note on synthetic speech

Synthetic audio inherits the TTS model's accent and pronunciation errors, and a model trained
only on it learns those too. Listen to a sample before training, and mix in real recordings from
[afrispeech-selector](https://github.com/AfriSpeech/afrispeech-selector) where they exist. The
generated dataset card says plainly that the audio is model-generated — leave that in.

---

## Development

```bash
git clone https://github.com/AfriSpeech/afrispeech-synth
cd afrispeech-synth
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite runs the whole pipeline end to end against a mock backend, so it needs
no API key and no network.

## Acknowledgements

- **[Google Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation)** generates the
  audio. Clips produced with it are subject to
  [Google's API terms](https://ai.google.dev/gemini-api/terms); check them before publishing a
  dataset, and say plainly in the dataset card that the audio is model-generated — the generated
  card does.
- **[africa-g2p](https://github.com/AfriSpeech/africa-g2p)**, built on Hartell's
  *Alphabets of Africa* (UNESCO, 1993) and Omniglot, for phonemisation.
- **[africa-corpus-builder](https://github.com/AfriSpeech/africa-corpus-builder)** and
  **[afriso](https://github.com/AfriSpeech/afriso)** (SIL ISO 639-3 + Glottolog) for text and
  language metadata.

## License

MIT — the pipeline. Generated audio is yours subject to your TTS provider's terms, and source
text keeps the licence of its own corpus.
