"""Ranking metrics on stable documentation section IDs, not chunk IDs."""


def unique_sections(section_ids: list[str]) -> list[str]:
    """Keep the first (highest ranked) occurrence of each source section."""
    return list(dict.fromkeys(section_ids))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("Recall requires at least one relevant section")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("MRR requires at least one relevant section")
    return next((1 / rank for rank, section_id in enumerate(ranked[:k], 1)
                 if section_id in relevant), 0.0)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("Hit rate requires at least one relevant section")
    return float(bool(set(ranked[:k]) & relevant))
