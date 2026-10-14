"""Multi-model fusion orchestration (personal baseline + optional diagnosis)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fusion.gating import AlertTier, FusionInput, classify_alert
from preprocessing.tag_schema import qc_veto, resolve_activity_segment


@dataclass(frozen=True)
class BeatFusionResult:
    qc_pass: bool
    qc_veto_reason: str | None
    activity_segment: str | None
    reconstruction_error: float | None
    alert_threshold: float | None
    smoothed_error: float | None
    instant_above: bool
    sustained_above: bool
    alert_tier: str
    diagnosis_suspicious: bool | None
    model_slot: str

    def to_dict(self) -> dict:
        return asdict(self)


def fuse_personal_beat(
    *,
    tags: frozenset[str],
    reconstruction_error: float,
    smoothed_error: float,
    alert_threshold: float,
    sustained_personal_alert: bool,
    diagnosis_suspicious: bool | None = None,
) -> BeatFusionResult:
    qc_pass, veto = qc_veto(tags)
    segment = resolve_activity_segment(tags)

    if not qc_pass:
        return BeatFusionResult(
            qc_pass=False,
            qc_veto_reason=veto,
            activity_segment=segment,
            reconstruction_error=reconstruction_error,
            alert_threshold=alert_threshold,
            smoothed_error=smoothed_error,
            instant_above=False,
            sustained_above=False,
            alert_tier=AlertTier.REJECT.value,
            diagnosis_suspicious=diagnosis_suspicious,
            model_slot="none",
        )

    if segment is None:
        return BeatFusionResult(
            qc_pass=True,
            qc_veto_reason=None,
            activity_segment=None,
            reconstruction_error=reconstruction_error,
            alert_threshold=alert_threshold,
            smoothed_error=smoothed_error,
            instant_above=False,
            sustained_above=False,
            alert_tier=AlertTier.STABLE.value,
            diagnosis_suspicious=diagnosis_suspicious,
            model_slot="unrouted",
        )

    instant = reconstruction_error > alert_threshold
    tier = classify_alert(
        FusionInput(
            qc_pass=True,
            reconstruction_error=reconstruction_error,
            alert_threshold=alert_threshold,
            diagnosis_suspicious=diagnosis_suspicious,
            sustained_personal_alert=sustained_personal_alert,
        )
    )

    return BeatFusionResult(
        qc_pass=True,
        qc_veto_reason=None,
        activity_segment=segment,
        reconstruction_error=reconstruction_error,
        alert_threshold=alert_threshold,
        smoothed_error=smoothed_error,
        instant_above=instant,
        sustained_above=sustained_personal_alert,
        alert_tier=tier.value,
        diagnosis_suspicious=diagnosis_suspicious,
        model_slot=f"personal_ae:{segment}",
    )
