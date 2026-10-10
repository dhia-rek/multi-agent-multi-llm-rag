import pytest

from src.agents.base import parse_json_safely


def test_plain_json():
    assert parse_json_safely('{"a": 1}') == {"a": 1}


def test_plain_json_list():
    assert parse_json_safely("[1, 2]") == [1, 2]


def test_fenced_json():
    assert parse_json_safely('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_fence_without_language():
    assert parse_json_safely('```\n{"a": 1}\n```') == {"a": 1}


def test_chatter_around_json():
    assert parse_json_safely('Sure! Here you go: {"a": {"b": 2}} Hope it helps.') == {"a": {"b": 2}}


def test_chatter_around_fence():
    assert parse_json_safely('Result:\n```json\n{"ok": true}\n```\nBye') == {"ok": True}


@pytest.mark.parametrize("raw", ["not json at all", "{broken: json}", "{'a': 1}"])
def test_invalid_returns_none(raw):
    assert parse_json_safely(raw) is None


@pytest.mark.parametrize("raw", ["", "   \n", None])
def test_empty_returns_none(raw):
    assert parse_json_safely(raw) is None
