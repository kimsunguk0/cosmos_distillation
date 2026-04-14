import numpy as np

from src.data.local_dataset import localize_trajectory


def test_localize_trajectory_uses_explicit_anchor_pose() -> None:
    xyz_world = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    quat_world = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    xyz_local, rot_local = localize_trajectory(
        xyz_world,
        quat_world,
        anchor_xyz_world=xyz_world[1],
        anchor_quat_xyzw=quat_world[1],
    )
    np.testing.assert_allclose(xyz_local[1], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(rot_local[1], np.eye(3), atol=1e-6)
