import hashlib


def compute_hash(text: str) -> str:

    normalized = " ".join(text.strip().split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()