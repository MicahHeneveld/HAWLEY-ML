"""Multi-model alert fusion (QC veto + personal error + optional diagnosis)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertTier(str, Enum):
    STABLE = "stable"
    PERSONAL_ANOMALY = "personal_anomaly"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"
    REJECT = "reject"


@dataclass(frozen=True)
class FusionInput:
    qc_pass: bool
    reconstruction_error: float
    alert_threshold: float
    diagnosis_suspicious: bool | None = None
    sustained_personal_alert: bool = False


def classify_alert(state: FusionInput) -> AlertTier:
    if not state.qc_pass:
        return AlertTier.REJECT

    personal_high = state.sustained_personal_alert or (
        state.reconstruction_error > state.alert_threshold
    )
    diagnosis = state.diagnosis_suspicious

    if diagnosis is None:
        return AlertTier.PERSONAL_ANOMALY if personal_high else AlertTier.STABLE

    if diagnosis and personal_high:
        return AlertTier.CRITICAL
    if diagnosis and not personal_high:
        return AlertTier.SUSPICIOUS
    if personal_high:
        return AlertTier.PERSONAL_ANOMALY
    return AlertTier.STABLE
