from pathlib import Path


def test_expected_top_level_paths_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "configs").exists()
    assert (root / "scripts").exists()
    assert (root / "src").exists()
