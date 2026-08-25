from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_skin_finish_human_review as review  # noqa: E402
import analyze_skin_finish_human_review as analyze  # noqa: E402


def _probe(*, width: int = 960, frames: int = 69) -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": 704,
                "nb_frames": str(frames),
                "r_frame_rate": "24/1",
                "duration": "2.875000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "32000",
                "channels": 2,
            },
        ]
    }


def test_video_pair_contract_requires_exact_review_geometry_and_timing():
    contract = review._validate_pair(_probe(), _probe())
    assert contract["width"] == 960
    assert contract["frame_count"] == 69
    assert contract["audio_present"] is True
    with pytest.raises(ValueError, match="contract mismatch"):
        review._validate_pair(_probe(), _probe(width=928))
    with pytest.raises(ValueError, match="contract mismatch"):
        review._validate_pair(_probe(), _probe(frames=68))


def test_blind_mapping_is_bijective_and_never_exposes_a_third_role():
    for _ in range(20):
        mapping = review._blind_mapping()
        assert set(mapping) == {"A", "B"}
        assert set(mapping.values()) == {"source", "candidate"}


def test_rendered_review_has_sync_zoom_abstain_and_export_without_method_leak():
    public = {
        "schema": review.SCHEMA,
        "review_id": "review-test",
        "created_at": "2026-08-25T00:00:00Z",
        "contract": review._video_contract(_probe()),
        "criteria": [{"id": key, "label": label} for key, label in review.CRITERIA],
        "boundary": "test",
        "sha256": "A" * 64,
    }
    html = review._render_html(public)
    assert "media/A.mp4" in html and "media/B.mp4" in html
    assert "poster/A.png" in html and "poster/B.png" in html
    assert "上一帧" in html and "下一帧" in html
    assert "放大" in html and "无法判断" in html
    assert "temporal_flicker" in html and "cross_person_spill" in html
    assert "导出评审JSON" in html
    assert "本组选择了“可判断”，请完成全部逐项选择" in html
    assert "missing.forEach(key=>criteria[key]='abstain')" in html
    assert 'name="failure_a"' in html and 'name="failure_b"' in html
    assert "hard_failures:{A:" in html
    assert "absolute" not in html.lower()
    assert "per_person_profile_live_00001" not in html


def test_public_manifest_hash_is_canonical_and_order_independent():
    left = {"schema": review.SCHEMA, "review_id": "x", "contract": {"a": 1}}
    right = {"contract": {"a": 1}, "review_id": "x", "schema": review.SCHEMA}
    assert review._hash_json(left) == review._hash_json(right)


def _review_contract():
    public = {
        "schema": review.SCHEMA,
        "review_id": "fixed-review",
        "created_at": "2026-08-25T00:00:00Z",
        "contract": review._video_contract(_probe()),
        "criteria": [{"id": key, "label": label} for key, label in review.CRITERIA],
        "boundary": "test",
    }
    public["sha256"] = review._hash_json(public)
    private = {
        "schema": f"{review.SCHEMA}/private-key",
        "review_id": public["review_id"],
        "public_manifest_sha256": public["sha256"],
        "mapping": {"A": "candidate", "B": "source"},
        "source": {"path": "source.mp4", "sha256": "1" * 64},
        "candidate": {"path": "candidate.mp4", "sha256": "2" * 64},
        "copied_media": {},
    }
    private["sha256"] = review._hash_json(private)
    submission = {
        "schema": review.SCHEMA,
        "review_id": public["review_id"],
        "public_manifest_sha256": public["sha256"],
        "assessability": "assessable",
        "criteria": {key: "A" for key, _ in review.CRITERIA},
        "hard_failures": {"A": [], "B": []},
        "left_notes": "left",
        "right_notes": "right",
        "notes": "overall",
    }
    return submission, public, private


def test_analysis_reveals_votes_but_never_auto_accepts_one_review():
    submission, public, private = _review_contract()
    report = analyze.analyze_review(submission, public, private)
    assert report["status"] == "HUMAN_REVIEW_COMPLETE"
    assert report["vote_counts"]["candidate"] == len(review.CRITERIA)
    assert set(report["revealed_votes"].values()) == {"candidate"}
    assert report["recommendation"] == "HUMAN_JUDGMENT_RECORDED_NO_AUTO_ACCEPT"
    assert report["automatic_accept"] is False


def test_analysis_rejects_candidate_hard_failure_and_honours_abstention():
    submission, public, private = _review_contract()
    submission["hard_failures"]["A"] = ["mouth_eye", "flicker"]
    hard = analyze.analyze_review(submission, public, private)
    assert hard["candidate_hard_fail"] is True
    assert hard["recommendation"] == "REJECT_CANDIDATE"
    submission["assessability"] = "source_insufficient"
    abstain = analyze.analyze_review(submission, public, private)
    assert abstain["status"] == "ABSTAIN_SOURCE_INSUFFICIENT"
    assert abstain["recommendation"] == "NO_AESTHETIC_DECISION"


def test_non_assessable_review_normalizes_unanswered_criterion_to_abstain():
    submission, public, private = _review_contract()
    submission["assessability"] = "unsure"
    submission["criteria"]["skin_naturalness"] = None
    report = analyze.analyze_review(submission, public, private)
    assert report["status"] == "ABSTAIN_UNSURE"
    assert report["recommendation"] == "NO_AESTHETIC_DECISION"
    assert report["revealed_votes"]["skin_naturalness"] == "abstain"
    assert report["normalized_unanswered_criteria"] == ["skin_naturalness"]
    assert report["automatic_accept"] is False


def test_analysis_fails_closed_on_manifest_or_vote_tampering():
    submission, public, private = _review_contract()
    submission["criteria"]["overall"] = None
    with pytest.raises(ValueError, match="invalid or missing votes"):
        analyze.analyze_review(submission, public, private)
    submission, public, private = _review_contract()
    submission["assessability"] = "unsure"
    submission["criteria"]["overall"] = "candidate"
    with pytest.raises(ValueError, match="invalid or missing votes"):
        analyze.analyze_review(submission, public, private)
    submission, public, private = _review_contract()
    public["contract"]["width"] = 1
    with pytest.raises(ValueError, match="manifest hash"):
        analyze.analyze_review(submission, public, private)
