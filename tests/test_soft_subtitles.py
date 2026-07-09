from app.soft_subtitles import SubtitleStream, select_text_subtitle_stream, srt_to_plain_text


def test_srt_to_plain_text_removes_indexes_timestamps_and_tags() -> None:
    content = """1
00:00:00,000 --> 00:00:01,000
<font color="white">你好，世界</font>

2
00:00:02,000 --> 00:00:03,000
Second line
"""

    assert srt_to_plain_text(content) == "你好，世界\n\nSecond line\n"


def test_select_text_subtitle_stream_prefers_chinese_text_track() -> None:
    streams = [
        SubtitleStream(subtitle_index=0, stream_id="0:2", codec="hdmv_pgs_subtitle", language="eng"),
        SubtitleStream(subtitle_index=1, stream_id="0:3", codec="subrip", language="eng"),
        SubtitleStream(subtitle_index=2, stream_id="0:4", codec="mov_text", language="zho"),
    ]

    selected = select_text_subtitle_stream(streams)

    assert selected is not None
    assert selected.subtitle_index == 2


def test_select_text_subtitle_stream_returns_none_without_text_track() -> None:
    streams = [SubtitleStream(subtitle_index=0, stream_id="0:2", codec="hdmv_pgs_subtitle", language="eng")]

    assert select_text_subtitle_stream(streams) is None
