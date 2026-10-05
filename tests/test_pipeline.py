import json
import os

import pytest

import mock_backend  # noqa: F401  (registers the "mock" backend)
from afrispeech_synth import RunConfig, from_dict, resolve
from afrispeech_synth.lang import Language
from afrispeech_synth.normalise import Normaliser
from afrispeech_synth.select import run as select_run, word_units
from afrispeech_synth.sources import _parse, clean, collect, split_sentences
from afrispeech_synth.tts.base import pcm_to_wav, wav_sample_rate
from afrispeech_synth import pipeline

SENTENCES = [
    "Akwaaba mo nyinaa wo Ghana ha",
    "Me din de Kofi na mefiri Kumasi",
    "Wo ho te sen anopa yi",
    "Ɔkɔɔ sukuu no mu ntɛm ara",
    "Akwaaba mo nyinaa wo Ghana ha",
]


# --------------------------------------------------------------- language

def test_resolve_by_code_and_name():
    assert resolve("twi").g2p_code == "twi"
    assert resolve("Twi").code == "twi"
    assert resolve("Yoruba").g2p_code == "yor"


def test_resolve_unknown_has_no_g2p():
    assert resolve("definitelynotalanguage").has_g2p is False


# --------------------------------------------------------------- sources

def test_parse_source_uris():
    assert _parse("corpus:twi@5000") == ("corpus", "twi", None, "5000")
    assert _parse("hf:org/ds#transcript@train") == ("hf", "org/ds", "transcript", "train")
    assert _parse("file:a.csv#text") == ("file", "a.csv", "text", None)


def test_clean_and_split():
    assert clean('  "hello   world" ') == "hello   world".replace("   ", " ")
    assert split_sentences(["One. Two! Three?"]) == ["One.", "Two!", "Three?"]


def test_collect_dedupes_in_order(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("\n".join(["b", "a", "b"]), encoding="utf-8")
    out = collect([f"file:{path}"], resolve("twi"))
    assert out == ["b", "a"]


# --------------------------------------------------------------- selection

def test_word_cover_is_minimal_and_complete():
    selection = select_run(SENTENCES, cover="word", min_chars=5)
    assert selection.coverage == 1.0
    covered = set()
    for sentence in selection.sentences:
        covered |= set(word_units(sentence))
    assert covered == set(word_units(" ".join(SENTENCES)))
    # the duplicate sentence must not be selected twice
    assert len(selection.sentences) == len(set(selection.sentences))


def test_phoneme_cover_uses_g2p():
    normaliser = Normaliser(resolve("twi"), "grapheme")
    selection = select_run(SENTENCES, cover="phoneme", unit_fn=normaliser.units, min_chars=5)
    assert selection.coverage == 1.0
    assert selection.total_units > 0


def test_length_filter():
    selection = select_run(["short", "a long enough sentence here"], cover="none", min_chars=10)
    assert selection.sentences == ["a long enough sentence here"]


def test_max_sentences_caps_selection():
    selection = select_run(SENTENCES, cover="word", min_chars=5, max_sentences=1)
    assert len(selection.sentences) == 1


# --------------------------------------------------------------- normalisation

def test_grapheme_and_ipa_differ():
    grapheme = Normaliser(resolve("twi"), "grapheme")("Akwaaba")
    ipa = Normaliser(resolve("twi"), "ipa")("Akwaaba")
    assert grapheme and ipa and grapheme != ipa


def test_normalise_none_is_passthrough():
    normaliser = Normaliser(resolve("twi"), "none")
    assert normaliser("Akwaaba") == "Akwaaba"
    assert normaliser.enabled is False


def test_missing_g2p_table_is_a_clear_error():
    from afrispeech_synth.lang import LanguageNotSupported
    with pytest.raises(LanguageNotSupported):
        Normaliser(Language(token="x", code="xxx", name="X"), "grapheme")


# --------------------------------------------------------------- audio

def test_pcm_to_wav_reads_rate_from_mime():
    wav = pcm_to_wav(b"\x00\x00" * 100, "audio/L16;rate=16000")
    assert wav[:4] == b"RIFF"
    assert wav_sample_rate(wav) == 16000


# --------------------------------------------------------------- config

def test_config_roundtrip_and_unknown_key():
    config = from_dict({"language": "twi", "select": {"cover": "word"}})
    assert config.select.cover == "word"
    with pytest.raises(ValueError):
        from_dict({"nonsense": 1})


# --------------------------------------------------------------- end to end

def _config(tmp_path, sentences_file):
    return from_dict({
        "language": "twi",
        "sources": [f"file:{sentences_file}"],
        "normalise": "grapheme",
        "out": str(tmp_path / "out"),
        "select": {"cover": "word", "min_chars": 5},
        "tts": {"backend": "mock", "model": "mock-1", "voice": "A",
                "rpm": 0, "concurrency": 4},
        "package": {"formats": ["parquet", "ljspeech"]},
    })


def test_end_to_end(tmp_path):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)

    result = pipeline.run(config)
    assert result["clips"] >= 4

    # Parquet shards carry embedded audio and the metadata the Hub reads to know
    # the column is audio. Checked with pyarrow rather than by decoding: decoding
    # pulls in an audio codec whose identity changes between datasets releases
    # (soundfile on 2.x, torchcodec on 5.x), and none of it is needed to verify
    # what we wrote.
    import json

    import pyarrow.parquet as pq

    shard = sorted((tmp_path / "out" / "data").glob("*.parquet"))[0]
    table = pq.read_table(shard)
    assert table.column_names == ["audio", "text", "normalised_text", "voice"]

    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["audio"] == {"sampling_rate": 24000, "_type": "Audio"}

    rows = table.to_pylist()
    assert rows[0]["audio"]["bytes"][:4] == b"RIFF"
    assert rows[0]["audio"]["path"].endswith(".wav")
    assert rows[0]["normalised_text"] != rows[0]["text"]
    # One voice per dataset: every clip is the voice the run configured.
    assert {r["voice"] for r in rows} == {"A"}

    # manifest and card
    manifest = (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == result["clips"]
    assert json.loads(manifest[0])["shard"].startswith("data/train-")
    card = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "africa-g2p" in card and "Twi" in card

    # ljspeech export
    assert (tmp_path / "out" / "ljspeech" / "metadata.csv").exists()


def test_resume_skips_finished_clips(tmp_path, capsys):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)

    pipeline.run(config)
    capsys.readouterr()
    pipeline.run(config)
    out = capsys.readouterr().out
    assert "already done" in out


def test_dry_run_calls_no_backend(tmp_path):
    source = tmp_path / "sentences.txt"
    source.write_text("\n".join(SENTENCES), encoding="utf-8")
    config = _config(tmp_path, source)
    config.tts.backend = "does-not-exist"
    result = pipeline.run(config, dry_run=True)
    assert result["dry_run"] is True
    assert not os.path.exists(os.path.join(config.work_dir, "audio", "000"))


def test_macrolanguage_falls_back_to_a_member():
    """`Akan` has no g2p table; Twi, one of its members, does."""
    pytest.importorskip("afriso")
    language = resolve("Akan")
    assert language.g2p_code == "twi"
    assert language.name == "Twi"


# --------------------------------------------------------------- coverage & voices

def test_voice_spread_is_even_and_uses_every_voice():
    from afrispeech_synth import voices

    assignment = voices.spread([f"l{i:03d}" for i in range(215)])
    counts = voices.distribution(assignment)
    assert len(counts) == len(voices.ALL) == 30
    assert max(counts.values()) - min(counts.values()) <= 1


def test_unknown_voice_is_caught():
    from afrispeech_synth import voices
    assert voices.unknown(["Zephyr", "Nope"]) == ["Nope"]


def test_coverage_reports_three_tiers():
    from afrispeech_synth import coverage

    catalogue = coverage.load()
    counts = catalogue.counts()
    # Registry keys are a lower bound, not the count: languages that reach a
    # table through an alias convert to universal too and must be included.
    assert counts["g2p"] >= 400
    assert counts["ready"] <= counts["g2p"]
    twi = catalogue.entries["twi"]
    assert twi.has_g2p and twi.status in {"ready", "bring text"}
    # Akan resolves by alias; matching registry keys alone used to drop it.
    assert catalogue.entries["aka"].has_g2p


def test_coverage_status_labels():
    from afrispeech_synth.coverage import Entry

    assert Entry("x", "X", has_g2p=True, has_text=True).status == "ready"
    assert Entry("x", "X", has_g2p=True).status == "bring text"
    assert Entry("x", "X", has_text=True).status == "no g2p"


# --------------------------------------------------------------- samples gallery

def _sample(code, name, voice, region="West Africa", text=None):
    from afrispeech_synth.samples import Sample
    text = text or "Akwaaba mo nyinaa wo ha"
    return Sample(code=code, name=name, family="Atlantic-Congo", region=region,
                  voice=voice, text=text, normalised_text=text.lower(),
                  audio=f"audio/{code}/{voice}.mp3")


def test_sample_sentence_pick_respects_length_window():
    from afrispeech_synth.samples import SAMPLE_MAX_CHARS, SAMPLE_MIN_CHARS, _pick_sentence

    short, good, long = "too short", "x" * (SAMPLE_MIN_CHARS + 5), "y" * (SAMPLE_MAX_CHARS + 50)
    assert _pick_sentence([short, long, good]) == good
    assert _pick_sentence([short, long]) is None


def test_card_opens_on_the_voice_it_shows_text_for():
    """The opening clip and the opening sentence must be the same clip.

    The card plays whichever voice the picker selects — the first option, so the
    first voice alphabetically — while the text came from whatever the manifest
    listed first. Where those differed the page opened showing one clip's
    sentence beside another clip's audio.
    """
    from afrispeech_synth import space

    # Manifest order deliberately unsorted: Zephyr first, Achird second.
    samples = [_sample("twi", "Twi", "Zephyr", text="zephyr sentence"),
               _sample("twi", "Twi", "Achird", text="achird sentence")]
    page = space.render(samples)

    assert "achird sentence" in page      # Achird sorts first, so it opens
    assert "zephyr sentence" not in page.split("data-texts")[0]


def test_gallery_page_renders_every_sample():
    from afrispeech_synth import space

    samples = [_sample("twi", "Twi", "Zephyr"),
               _sample("yor", "Yoruba", "Puck"),
               _sample("swh", "Swahili", "Kore", region="East Africa")]
    page = space.render(samples)

    assert page.count('<article class="card"') == 3
    for sample in samples:
        # Clips are addressed by rule — <base>/audio/<code>/<voice><ext> — so the
        # page carries the code and the voice, not 17,000 spelled-out URLs.
        assert f'data-code="{sample.code}"' in page
        assert sample.voice in page
    assert "function clipUrl(" in page
    # The extension travels with the rule: --compress writes .mp3, not .wav.
    assert 'AUDIO_EXT = ".mp3"' in page
    # provenance the page must not quietly drop
    assert "Gemini TTS" in page and "machine-generated" in page
    assert "Bible translations" in page
    assert "africa-g2p" in page


def test_gallery_escapes_text():
    from afrispeech_synth import space

    sample = _sample("twi", "Twi & <b>friends</b>", "Zephyr")
    sample.text = '<script>alert("x")</script>'
    page = space.render([sample])
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_gallery_build_writes_space_files(tmp_path):
    from afrispeech_synth import space

    space.build([_sample("twi", "Twi", "Zephyr")], str(tmp_path))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---") and "sdk: static" in readme
    assert (tmp_path / "index.html").exists()
    assert json.loads((tmp_path / "samples.json").read_text(encoding="utf-8"))[0]["code"] == "twi"


def test_env_file_does_not_override_real_environment(tmp_path, monkeypatch):
    from afrispeech_synth.env import load

    (tmp_path / ".env").write_text('export GEMINI_API_KEY="from-file"\nOTHER=x\n',
                                   encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "already-exported")
    load()
    assert os.environ["GEMINI_API_KEY"] == "already-exported"
    assert os.environ["OTHER"] == "x"


def test_universal_mode_uses_the_shared_letter_set():
    """Twi ɔ/ɛ collapse to o/e; grapheme mode keeps them."""
    grapheme = Normaliser(resolve("twi"), "grapheme")("Na hɔ bɔbea ɛyɛ")
    universal = Normaliser(resolve("twi"), "universal")("Na hɔ bɔbea ɛyɛ")
    assert "ɔ" in grapheme and "ɔ" not in universal
    assert "ho" in universal and "bobea" in universal


def test_universal_coverage_units_stay_language_units():
    """Universal respells sounds, so cover still counts the language's own units."""
    normaliser = Normaliser(resolve("twi"), "universal")
    assert normaliser.enabled
    assert normaliser.units("Akwaaba")[:3] == ["a", "kw", "a"]


def test_universal_is_the_default_mode():
    from afrispeech_synth import RunConfig
    assert RunConfig().normalise == "universal"


def test_every_non_default_mode_carries_a_warning():
    from afrispeech_synth.normalise import MODES, MODE_WARNINGS
    assert set(MODE_WARNINGS) == set(MODES) - {"universal"}


# --------------------------------------------------------------- punctuation

def test_only_phrasing_punctuation_survives():
    from afrispeech_synth.normalise import strip_punctuation

    assert strip_punctuation("Hello, world! Is it ok? Yes.") == "Hello, world! Is it ok? Yes."
    assert strip_punctuation("a 'b' c") == "a b c"
    assert strip_punctuation("anan-mo") == "ananmo"      # joins, never splits a word
    assert strip_punctuation("*Abalahamun") == "Abalahamun"
    assert strip_punctuation("“curly” — dash; colon:") == "“curly” dash; colon"
    assert strip_punctuation('He said "come here"; then left.') == 'He said "come here"; then left.'
    assert strip_punctuation("She said “yes” , then left .") == "She said “yes”, then left."


def test_stripping_leaves_no_stray_spaces():
    from afrispeech_synth.normalise import strip_punctuation
    assert strip_punctuation("word ' , next") == "word, next"


def test_normaliser_strips_source_punctuation():
    """Anyin writes apostrophes; a synthesiser reads them as pauses."""
    source = "Ɛ 'nwun Zozi, anan-mɔ dunman'n."
    out = Normaliser(resolve("any"), "universal")(source)
    assert "'" not in out and "-" not in out
    assert out.endswith(".") and "," in out


def test_ipa_modifier_letters_are_not_punctuation():
    """ʼ and ǁ are letters in Unicode, and IPA output depends on them."""
    out = Normaliser(resolve("xho"), "ipa")("ukuba xa")
    assert "ʼ" in out      # ejective mark
    assert "ǁ" in out      # lateral click


def test_invisible_characters_are_removed():
    """Zero-width spaces sit inside words and pass every visible check."""
    from afrispeech_synth.normalise import strip_punctuation

    assert strip_punctuation("nebe\u200brechi") == "neberechi"
    assert strip_punctuation("a\u200db\ufeffc") == "abc"


def test_multi_key_config_and_scaling(monkeypatch):
    """Multiple API keys scale concurrency and RPM automatically."""
    monkeypatch.setenv("GEMINI_API_KEYS", "key1,key2,key3")
    cfg = RunConfig(language="twi")
    assert cfg.tts.api_keys_env == "GEMINI_API_KEYS"
    assert cfg.tts.scale_concurrency_with_keys is True
