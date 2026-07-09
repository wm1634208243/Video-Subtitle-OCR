from app.subtitles import SubtitleEntry, clean_text, merge_ocr_results, normalize, srt_timestamp, write_outputs


def test_clean_text_keeps_useful_subtitle_text() -> None:
    assert clean_text("  A classic doesn't have to be a relic.  ", min_length=2) == (
        "A classic doesn't have to be a relic."
    )
    assert clean_text("  这是一句中文字幕。  ", min_length=2) == "这是一句中文字幕。"


def test_clean_text_rejects_short_or_noisy_text() -> None:
    assert clean_text("A", min_length=2) is None
    assert clean_text("□□□□□□", min_length=2) is None


def test_normalize_ignores_common_punctuation() -> None:
    assert normalize("Hello, World!") == "helloworld"
    assert normalize("你好，世界！") == "你好世界"


def test_merge_ocr_results_merges_similar_neighbors() -> None:
    samples = [
        (0.0, "Hello world"),
        (0.5, "Hello, world!"),
        (1.0, ""),
        (1.5, "Next line"),
    ]

    entries = merge_ocr_results(samples, fps=2.0, min_length=2, similarity=0.75)

    assert len(entries) == 2
    assert entries[0] == SubtitleEntry(text="Hello world", start=0.0, end=1.0)
    assert entries[1] == SubtitleEntry(text="Next line", start=1.5, end=2.0)


def test_merge_ocr_results_drops_contained_tail_fragment() -> None:
    samples = [
        (0.0, "all driving scenarios and all climate conditions."),
        (1.0, ""),
        (2.0, "scenarios and all climate conditions."),
    ]

    entries = merge_ocr_results(samples, fps=1.0, min_length=2, similarity=0.95)

    assert entries == [
        SubtitleEntry(text="all driving scenarios and all climate conditions.", start=0.0, end=3.0)
    ]


def test_srt_timestamp_rounds_to_milliseconds() -> None:
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(3661.2345) == "01:01:01,234"


def test_write_outputs(tmp_path) -> None:
    srt_path = tmp_path / "out.srt"
    txt_path = tmp_path / "out.txt"
    write_outputs([SubtitleEntry(text="Hello", start=0, end=1.25)], srt_path, txt_path)

    assert srt_path.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,250\nHello\n"
    assert txt_path.read_text(encoding="utf-8") == "Hello\n"
