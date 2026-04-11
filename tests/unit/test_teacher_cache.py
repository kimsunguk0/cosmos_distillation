import json
from pathlib import Path

from src.data.teacher_cache import inspect_teacher_sample


def test_inspect_teacher_sample_detects_missing_canonical(tmp_path: Path) -> None:
    readiness = inspect_teacher_sample("sample_x", tmp_path)
    assert readiness.status == "awaiting_canonical_sample"
    assert "canonical_sample_missing" in readiness.blockers


def test_inspect_teacher_sample_marks_blocked_without_frames(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_y"
    sample_dir.mkdir(parents=True)
    (sample_dir / "ego_history_xyz.npy").write_bytes(b"0")
    (sample_dir / "ego_history_rot.npy").write_bytes(b"0")
    (sample_dir / "ego_future_xyz.npy").write_bytes(b"0")
    (sample_dir / "ego_future_rot.npy").write_bytes(b"0")
    (sample_dir / "rel_timestamps.json").write_text("{}", encoding="utf-8")
    (sample_dir / "abs_timestamps.json").write_text("{}", encoding="utf-8")
    (sample_dir / "sample_meta.json").write_text(
        json.dumps(
            {
                "frame_offsets_sec": [-0.3, -0.2, -0.1, 0.0],
                "decoder_status": {
                    "camera_cross_left_120fov": "skipped",
                    "camera_front_wide_120fov": "skipped",
                    "camera_cross_right_120fov": "skipped",
                    "camera_front_tele_30fov": "skipped",
                },
            }
        ),
        encoding="utf-8",
    )
    readiness = inspect_teacher_sample("sample_y", tmp_path)
    assert readiness.status == "blocked"
    assert "image_frames_missing" in readiness.blockers
