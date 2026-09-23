from selector_public_text_refs import deduplicate_public_text


def test_roundtrip_preserves_all_events_and_values():
    text = "unchanged observation " * 100
    packet = {"events": [{"id": "a", "content": text}, {"id": "b", "content": text}], "index": 0}
    result = deduplicate_public_text(packet)
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"public_text_ref"}:
                return result["public_texts"][value["public_text_ref"]]
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    assert expand(result["packet"]) == packet
    assert len(result["public_texts"]) == 1


def test_no_repetitions_keeps_original():
    packet = {"a": "x" * 700, "b": "y" * 700}
    assert deduplicate_public_text(packet) is packet
