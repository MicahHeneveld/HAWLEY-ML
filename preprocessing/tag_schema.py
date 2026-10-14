"""Sandbox tag normalization (schema drift across export batches)."""

from __future__ import annotations

# Layer A — kinematic context (tags_schema v2 + legacy exports)
LAYER_A_SEDENTARY = frozenset({"sedentary_stationary", "stationary_activity"})
LAYER_A_WALKING = frozenset({"walking"})
LAYER_A_CYCLING = frozenset({"cycling"})

LAYER_C_REJECT = frozenset(
    {"motion_artifact", "poor_contact", "baseline_wander"}
)


def normalize_tags(tags: frozenset[str] | set[str]) -> frozenset[str]:
    return frozenset(str(t).strip() for t in tags if str(t).strip())


def has_sedentary_context(tags: frozenset[str]) -> bool:
    return bool(normalize_tags(tags) & LAYER_A_SEDENTARY)


def qc_veto(tags: frozenset[str]) -> tuple[bool, str | None]:
    """Layer C veto — no alert / no training update when failed."""
    tags = normalize_tags(tags)
    for bad in LAYER_C_REJECT:
        if bad in tags:
            return False, bad
    return True, None


def match_layer_a(tags: frozenset[str], layer_a: str) -> bool:
    tags = normalize_tags(tags)
    if layer_a == "sedentary_stationary":
        return has_sedentary_context(tags)
    return layer_a in tags


def resolve_activity_segment(tags: frozenset[str]) -> str | None:
    """Route beat to personal baseline segment model."""
    tags = normalize_tags(tags)
    if "resting_hr" in tags and has_sedentary_context(tags):
        return "resting_hr"
    if has_sedentary_context(tags):
        return "rest"
    if tags & LAYER_A_WALKING:
        return "walk"
    if tags & LAYER_A_CYCLING:
        return "cycling"
    return None
