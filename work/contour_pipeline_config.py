"""Named, serializable configurations for continuous-contour processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ContourPipelineConfig:
    pitch_filter: str = "mean_hz_90"
    # The full-model contour is the displayed pitch source.  The tiny-model
    # lead contour controls voicing and has its own independently tuned filter.
    lead_pitch_filter: str = "mean_hz_90"
    rms_mode: str = "otsu"
    rms_otsu_multiplier: float = 1.25
    rms_min_class_fraction: float = 0.08
    rms_min_separability: float = 0.55
    rms_max_valley_ratio: float = 0.70
    rms_fallback_percentile: float = 20.0
    rms_fallback_multiplier: float = 0.75
    seed_confidence: float = 0.50
    minimum_seed_rate_hz: float = 0.0
    # Reject low-energy connected regions only when an independent estimator
    # is also absent. This catches long separator residue that a duration-only
    # rule cannot distinguish from a real short note.
    unconfirmed_rms_percentile: float = 0.0
    unconfirmed_max_secondary_fraction: float = 0.0
    unconfirmed_secondary_confidence: float = 0.50
    # Trim a weak prefix/suffix when it jumps to a different pitch while pYIN
    # still supports the stable interior. These are usually breaths or pitch
    # tracker dropouts attached to an otherwise valid phrase.
    edge_excursion_max_seconds: float = 0.0
    edge_excursion_context_seconds: float = 0.12
    edge_excursion_min_seconds: float = 0.03
    edge_excursion_min_pitch_difference: float = 6.0
    edge_excursion_expansion_difference: float = 3.0
    edge_excursion_max_rms_ratio: float = 0.45
    edge_excursion_secondary_confidence: float = 0.50
    edge_excursion_secondary_tolerance: float = 2.0
    edge_excursion_max_secondary_fraction: float = 0.15
    # Remove a whole short region only when its pitch is highly unstable and
    # neither CREPE nor pYIN provides sustained support. This targets inhaled
    # noise without treating a quiet, stable held note as unvoiced.
    unstable_segment_max_seconds: float = 0.0
    unstable_segment_min_pitch_span: float = 5.0
    unstable_segment_max_primary_seed_fraction: float = 0.80
    unstable_segment_max_secondary_fraction: float = 0.05
    unstable_segment_secondary_confidence: float = 0.50
    # Trim boundary frames that spike or ramp before the pitch settles.
    # Interior excursion repair needs context on both sides, so onset/offset
    # artifacts are structurally unreachable for it; the weak-edge rule only
    # fires on quiet edges >= 5 st away. Real approaches are protected twice:
    # a flat sustained pickup note is exempt (stability guard) and pitch that
    # pYIN or the unsmoothed CREPE output also tracks is exempt (evidence
    # guard). Only pYIN is independent; raw CREPE is corroborating same-model
    # evidence.
    unsettled_edge_max_seconds: float = 0.0
    unsettled_edge_context_seconds: float = 0.12
    unsettled_edge_settle_tolerance_st: float = 1.5
    unsettled_edge_min_deviation_st: float = 3.0
    unsettled_edge_min_prefix_span_st: float = 2.0
    unsettled_edge_short_prefix_seconds: float = 0.06
    unsettled_edge_secondary_confidence: float = 0.60
    unsettled_edge_raw_confidence: float = 0.50
    unsettled_edge_support_tolerance_st: float = 1.5
    unsettled_edge_max_support_fraction: float = 0.50
    # The product issue is a spurious lead-in before a sung note.  Offset
    # trimming is separately opt-in because short releases and grace notes
    # are common, and the first symmetric candidate lost too much recall on
    # separator stems.
    unsettled_edge_trim_offsets: bool = True
    # Experimental pitch-only alternative to deleting an unsettled onset.
    # For a voiced phrase preceded by a real gap, replace at most this much
    # leading pitch with high-confidence pYIN. Prefix mode stops when the
    # primary agrees; available mode substitutes each supported leading frame.
    # Stable, confident pickups remain protected in both modes.
    secondary_onset_repair_seconds: float = 0.0
    secondary_onset_mode: str = "prefix"
    secondary_onset_min_gap_seconds: float = 0.05
    secondary_onset_min_confidence: float = 0.60
    secondary_onset_min_deviation_st: float = 2.50
    secondary_onset_agreement_st: float = 1.00
    secondary_onset_evidence_frames: int = 3
    secondary_onset_primary_stability_frames: int = 3
    secondary_onset_stability_tolerance_st: float = 1.00
    secondary_onset_primary_confidence: float = 0.60
    secondary_seed_confidence: float = 0.75
    secondary_seed_fraction: float = 0.20
    max_jump_semitones: float = 0.80
    minimum_segment_seconds: float = 0.09
    evidence_aware_excursions: bool = False
    excursion_maximum_frames: int = 30
    note_region_source_tolerance_st: float = 3.0
    pitch_merge: str = "mixed_full"
    # Late-stage fallback that trusts lead-only pYIN exactly where the anomaly
    # miner would flag the exported contour: estimator disagreement, sharp
    # frame jumps, and low-support voicing. Confident pYIN replaces flagged
    # pitch, short flagged remainders are bridged between trusted neighbors,
    # and low-support regions that pYIN actively rejects are unvoiced. The
    # thresholds intentionally mirror prepare_contour_anomaly_review so the
    # stage acts on the same territory a human reviewer would have been shown.
    pyin_fallback_enabled: bool = False
    pyin_fallback_confidence: float = 0.55
    pyin_fallback_disagreement_st: float = 4.0
    pyin_fallback_min_disagreement_seconds: float = 0.04
    pyin_fallback_jump_st: float = 2.5
    pyin_fallback_jump_dilation_seconds: float = 0.04
    pyin_fallback_bridge_max_seconds: float = 0.15
    pyin_fallback_low_support_broad_confidence: float = 0.42
    pyin_fallback_low_support_lead_confidence: float = 0.48
    pyin_fallback_support_confidence: float = 0.35
    pyin_fallback_reject_confidence: float = 0.10
    pyin_fallback_min_low_support_seconds: float = 0.06
    pyin_fallback_max_unvoice_seconds: float = 1.0
    # The unvoice step is the only content-deleting behavior in the fallback
    # module; disable it to keep the pitch corrections (override/bridge/edge)
    # without removing any voicing.
    pyin_fallback_unvoice_enabled: bool = True
    # Segment-edge substitution: the first/last window of every voiced
    # segment takes confident pYIN directly, because CREPE plus smoothing
    # dominates exactly those settling frames. 50 ms and confidence 0.60
    # match the guardrail-validated pyin-onset-050-direct-all candidate.
    pyin_fallback_edge_seconds: float = 0.05
    pyin_fallback_edge_confidence: float = 0.60
    # Human-reviewed region models remain a separate, double-gated stage:
    # the preset must opt in and the serialized bundle must explicitly pass
    # its completion/diversity/grouped-validation readiness checks.
    reviewed_model_enabled: bool = False
    reviewed_model_path: str = (
        "experiments/reviewed_contour_models/model_bundle.json"
    )
    reviewed_model_require_production_eligible: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE_CONFIG = ContourPipelineConfig()
IMPROVED_CONFIG = ContourPipelineConfig(
    pitch_filter="median_midi_70_mean_midi_30",
    rms_mode="guarded_otsu",
    rms_fallback_multiplier=0.50,
    minimum_seed_rate_hz=0.50,
    evidence_aware_excursions=True,
    excursion_maximum_frames=20,
    pitch_merge="confidence_aware",
)
PRODUCTION_V1_CONFIG = replace(
    IMPROVED_CONFIG,
    # The guarded fallback improved recall on development data but exceeded
    # the frozen test's false-alarm guardrail. Keep the implementation and
    # preset for future work without silently shipping the failed threshold.
    rms_mode="otsu",
)
# The original v2 duration-only breath veto was reverted after it changed none
# of the user-reported defects. Keep the name as a compatibility alias while
# production-v3 applies the replacement evidence-aware cleanup below.
PRODUCTION_V2_CONFIG = PRODUCTION_V1_CONFIG
PRODUCTION_V3_CONFIG = replace(
    PRODUCTION_V1_CONFIG,
    unconfirmed_rms_percentile=10.0,
    unconfirmed_max_secondary_fraction=0.05,
    edge_excursion_max_seconds=0.40,
)
# v3's whole-region low-energy veto removed a confirmed quiet held note in
# Shout Baby. Production v4 keeps the edge-specific artifact repair and the
# octave-boundary fix, but disables that ambiguous sustained-region veto.
PRODUCTION_V4_CONFIG = replace(
    PRODUCTION_V1_CONFIG,
    edge_excursion_max_seconds=0.40,
)
PRODUCTION_V5_CONFIG = replace(
    PRODUCTION_V1_CONFIG,
    edge_excursion_max_seconds=0.80,
    edge_excursion_min_pitch_difference=5.0,
    unstable_segment_max_seconds=0.35,
)
PRODUCTION_V6_CONFIG = replace(
    PRODUCTION_V5_CONFIG,
    # Preserve the superseded configuration that was already shipped and
    # benchmarked under the production-v6 name. Preset names are immutable so
    # stored reports and exported artifacts always describe one exact chain.
    minimum_seed_rate_hz=0.0,
    unstable_segment_max_seconds=0.0,
)
PRODUCTION_V6R_CONFIG = replace(
    PRODUCTION_V5_CONFIG,
    # One-filter-off tests showed the duration-scaled seed rate changed no
    # frame on any isolated-vocal or commercial-separator clip, so it is
    # removed. The 350 ms unstable-region rule stays: disabling it re-voiced
    # confirmed Shout Baby inhales and raised separator-dev false alarms from
    # 12.68% to 12.83% for a development-only recall gain.
    minimum_seed_rate_hz=0.0,
)

PRESETS = {
    "baseline": BASELINE_CONFIG,
    "median90": replace(
        BASELINE_CONFIG,
        pitch_filter="median_midi_90",
        lead_pitch_filter="median_midi_90",
    ),
    "median70": replace(
        BASELINE_CONFIG,
        pitch_filter="median_midi_70",
        lead_pitch_filter="median_midi_70",
    ),
    "median70-mean30": replace(
        BASELINE_CONFIG,
        pitch_filter="median_midi_70_mean_midi_30",
        lead_pitch_filter="median_midi_70_mean_midi_30",
    ),
    "weighted-median70": replace(
        BASELINE_CONFIG,
        pitch_filter="weighted_median_midi_70",
        lead_pitch_filter="weighted_median_midi_70",
    ),
    "anchor-median70-mean30": replace(
        BASELINE_CONFIG,
        pitch_filter="median_midi_70_mean_midi_30",
    ),
    "guarded-rms-050": replace(
        BASELINE_CONFIG, rms_mode="guarded_otsu", rms_fallback_multiplier=0.50
    ),
    "guarded-rms-075": replace(
        BASELINE_CONFIG, rms_mode="guarded_otsu", rms_fallback_multiplier=0.75
    ),
    "guarded-rms-100": replace(
        BASELINE_CONFIG, rms_mode="guarded_otsu", rms_fallback_multiplier=1.00
    ),
    "protected-excursions": replace(
        BASELINE_CONFIG,
        evidence_aware_excursions=True,
        excursion_maximum_frames=20,
    ),
    "confidence-merge": replace(BASELINE_CONFIG, pitch_merge="confidence_aware"),
    "improved-candidate": IMPROVED_CONFIG,
    "improved-v1": IMPROVED_CONFIG,
    "production-v1": PRODUCTION_V1_CONFIG,
    "production-v2": PRODUCTION_V2_CONFIG,
    "production-v3": PRODUCTION_V3_CONFIG,
    "production-v4": PRODUCTION_V4_CONFIG,
    "production-v5": PRODUCTION_V5_CONFIG,
    "production-v6": PRODUCTION_V6_CONFIG,
    "production-v6r": PRODUCTION_V6R_CONFIG,
    # Onset-artifact candidates (July 2026): must pass the frozen regression
    # and the published-contour spot tests before any production flip.
    "unsettled-edges": replace(
        PRODUCTION_V6R_CONFIG, unsettled_edge_max_seconds=0.20
    ),
    "onset-settle-080": replace(
        PRODUCTION_V6R_CONFIG,
        unsettled_edge_max_seconds=0.08,
        unsettled_edge_trim_offsets=False,
    ),
    "onset-settle-120": replace(
        PRODUCTION_V6R_CONFIG,
        unsettled_edge_max_seconds=0.12,
        unsettled_edge_trim_offsets=False,
    ),
    "onset-settle-160": replace(
        PRODUCTION_V6R_CONFIG,
        unsettled_edge_max_seconds=0.16,
        unsettled_edge_trim_offsets=False,
    ),
    "lead-median": replace(
        PRODUCTION_V6R_CONFIG, lead_pitch_filter="median_midi_70_mean_midi_30"
    ),
    "onset-candidate": replace(
        PRODUCTION_V6R_CONFIG,
        unsettled_edge_max_seconds=0.20,
        lead_pitch_filter="median_midi_70_mean_midi_30",
    ),
    "pyin-onset-050": replace(
        PRODUCTION_V6R_CONFIG,
        secondary_onset_repair_seconds=0.05,
    ),
    "pyin-onset-050-direct": replace(
        PRODUCTION_V6R_CONFIG,
        secondary_onset_repair_seconds=0.05,
        secondary_onset_mode="available",
        secondary_onset_min_gap_seconds=0.01,
        secondary_onset_evidence_frames=1,
    ),
    "pyin-onset-050-direct-150": replace(
        PRODUCTION_V6R_CONFIG,
        secondary_onset_repair_seconds=0.05,
        secondary_onset_mode="available",
        secondary_onset_min_gap_seconds=0.01,
        secondary_onset_evidence_frames=1,
        secondary_onset_min_deviation_st=1.50,
    ),
    "pyin-onset-050-direct-all": replace(
        PRODUCTION_V6R_CONFIG,
        secondary_onset_repair_seconds=0.05,
        secondary_onset_mode="available",
        secondary_onset_min_gap_seconds=0.01,
        secondary_onset_evidence_frames=1,
        secondary_onset_min_deviation_st=0.0,
    ),
    # This preset is intentionally not selected by PRODUCTION_PRESET. It can
    # only run after the reviewed-label trainer promotes an eligible bundle.
    "reviewed-model-candidate": replace(
        PRODUCTION_V6R_CONFIG,
        reviewed_model_enabled=True,
    ),
    # Experimental (July 2026): trust lead-only pYIN wherever the anomaly
    # miner would flag the final contour. Not selected by PRODUCTION_PRESET;
    # exported per-song for review of gunjou/shoutbaby defects. Unsettled
    # edges are trimmed under this preset's core principle — an edge pitch
    # pYIN does not vouch for is not trusted — so the raw-CREPE branch of the
    # evidence guard is disabled by an unreachable confidence threshold and
    # pYIN is the only witness that can save a deviant edge.
    # A conservativeness sweep (gunjou+shoutbaby, 2026-07-18) showed the
    # 200 ms trim window captured most severe edge spikes but also produced
    # 130-200 ms removals; capping at 120 ms keeps every trim a short settling
    # sliver (<=110 ms observed) while still clearing gunjou's severe edge
    # spikes 21/15 -> 8/7. The min-deviation and raw-CREPE-guard knobs were
    # measured no-ops here (every trimmed edge deviates >=4 st and raw CREPE
    # never corroborates the settled level), but raw_confidence stays
    # unreachable so pYIN remains the sole witness the moment that changes.
    "pyin-fallback-flagged": replace(
        PRODUCTION_V6R_CONFIG,
        pyin_fallback_enabled=True,
        unsettled_edge_max_seconds=0.12,
        unsettled_edge_raw_confidence=1.01,
    ),
    # Correcting filters only: the fallback's pitch corrections (override,
    # bridge, edge substitution) with both new deletion behaviors off (the
    # unvoice step and the unsettled-edge trim). Keeps v6r's shipped
    # conservative prunes. No new content is removed vs v6r.
    "pyin-correct-only": replace(
        PRODUCTION_V6R_CONFIG,
        pyin_fallback_enabled=True,
        pyin_fallback_unvoice_enabled=False,
    ),
}

# Simplified production subset selected from isolated-vocal and commercial-
# separator ablations. The guarded RMS fallback stays experimental after
# failing the frozen-test false-alarm guardrail.
PRODUCTION_PRESET = "production-v6r"
PRODUCTION_CONFIG = PRESETS[PRODUCTION_PRESET]


def get_config(name: str) -> ContourPipelineConfig:
    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown contour preset {name!r}; choose from {', '.join(PRESETS)}"
        ) from error
