---
license: mit
task_categories:
- text-to-speech
- automatic-speech-recognition
pretty_name: Synthetic Voice Samples · Africa
size_categories:
- 10K<n<100K
tags:
- synthetic
- tts
- african-languages
- speech-synthesis
- multilingual
configs:
- config_name: default
  data_files: data/*.parquet
---

# Synthetic Voice Samples · Africa

**Synthetic speech. No human speaker was recorded for any clip in this dataset.**

Generated with [afrispeech-synth](https://github.com/AfriSpeech/afrispeech-synth): text from
[africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus), normalised to a
universal orthography with [africa-g2p](https://github.com/AfriSpeech/africa-g2p), spoken by
Google Gemini's Live API.

- **17,010 clips** · **38.8 hours** · **566 languages** · **30 voices**
- Every clip is a **distinct sentence** — no sentence is repeated
- Each language is read by up to 30 different voices, one sentence per voice
- ~1.29 hours per voice

## Audio

Exactly as the model produced it — no resampling, no re-encoding, no conversion.

| | |
|---|---|
| Format | WAV, 16-bit signed PCM |
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |
| Clip length | 8.2s mean, 1.4-27.1s |

## Two ways in

`data/*.parquet` carries the audio inline for training. `audio/` holds the same clips as
individual WAV files, which is what the browsable gallery streams and what you want if you
need one clip rather than the set. The clips are identical; only the packaging differs.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("AfriSpeech/multivoice-synthetic-speech", split="train")

twi = ds.filter(lambda r: r["language"] == "twi")     # one language
zephyr = ds.filter(lambda r: r["voice"] == "Zephyr")  # one voice
```

Rows are **shuffled**, so language and voice are spread across every shard: streaming the
first shard gives a cross-section rather than one language in one voice.

## Columns

| Column | |
|---|---|
| `audio` | the clip (24 kHz mono WAV) |
| `text` | the sentence as it appears in the corpus |
| `normalised_text` | what the model was actually asked to read (universal orthography) |
| `voice` | which Gemini voice spoke it |
| `language` | ISO 639-3 code |
| `language_name` | e.g. Dagbani |
| `family` | e.g. Niger-Congo |
| `region` | e.g. West Africa |

`text` and `normalised_text` differ because universal orthography maps a language's own
letters onto the shared set most African languages use — `ɔ` to `o`, `ɛ` to `e`, `gy` to `j`.
The model read `normalised_text`.

## What this is not

Recorded speech, a pronunciation reference, or evidence that a language *sounds* like this.
A synthetic clip is a model's guess at an orthography, and quality varies enormously by
language — these voices were built for widely-spoken languages and are being asked to read
hundreds of others. Treat it as a starting point for bootstrapping, never as ground truth.
