from src.cache.disk_cache import DiskCache


def test_roundtrip(tmp_path):
    cache = DiskCache("ns", root=tmp_path)
    cache.set({"q": 1}, {"answer": [1, 2]})
    assert cache.get({"q": 1}) == {"answer": [1, 2]}


def test_miss_returns_none(tmp_path):
    assert DiskCache("ns", root=tmp_path).get({"q": 1}) is None


def test_key_ignores_dict_order_but_not_content(tmp_path):
    cache = DiskCache("ns", root=tmp_path)
    assert cache._key({"a": 1, "b": 2}) == cache._key({"b": 2, "a": 1})
    assert cache._key({"a": 1}) != cache._key({"a": 2})


def test_different_payloads_do_not_collide(tmp_path):
    cache = DiskCache("ns", root=tmp_path)
    cache.set({"q": "a"}, "A")
    cache.set({"q": "b"}, "B")
    assert (cache.get({"q": "a"}), cache.get({"q": "b"})) == ("A", "B")
    assert len(list(cache.dir.glob("*.pkl"))) == 2


def test_clear(tmp_path):
    cache = DiskCache("ns", root=tmp_path)
    cache.set({"q": 1}, "A")
    cache.clear()
    assert cache.get({"q": 1}) is None
