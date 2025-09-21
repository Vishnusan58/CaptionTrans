import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import build_srt_from_segments, format_timestamp


def test_format_timestamp_basic():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.234) == "00:00:01,234"
    assert format_timestamp(3661.005) == "01:01:01,005"
    assert format_timestamp(-5) == "00:00:00,000"


def test_build_srt_from_segments():
    segments = [
        {"start": 0.0, "end": 1.5, "text": "Hello --> world"},
        {"start": 1.5, "end": 3.0, "text": "Second line"},
    ]
    expected = (
        "1\n00:00:00,000 --> 00:00:01,500\nHello → world\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nSecond line\n"
    )
    assert build_srt_from_segments(segments) == expected


def test_build_srt_from_empty_segments():
    assert build_srt_from_segments([]) == ""
