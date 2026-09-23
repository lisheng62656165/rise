"""Lossless deduplication of repeated public text in oversized packets."""
import json
from collections import Counter


def deduplicate_public_text(packet, minimum_chars=512):
    counts = Counter()

    def count(value):
        if isinstance(value, str) and len(value) >= minimum_chars:
            counts[value] += 1
        elif isinstance(value, dict):
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)

    count(packet)
    refs = {value: f"text-{i}" for i, (value, n) in enumerate(counts.items()) if n > 1}
    if not refs:
        return packet

    def replace(value):
        if isinstance(value, str) and value in refs:
            return {"public_text_ref": refs[value]}
        if isinstance(value, dict):
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value

    compact = {
        "encoding": "Repeated public_text_ref values expand to the exact text in public_texts. Every occurrence remains a separate event; repetitions are not removed.",
        "packet": replace(packet),
        "public_texts": {ref: value for value, ref in refs.items()},
    }
    return compact if len(json.dumps(compact)) < len(json.dumps(packet)) else packet
