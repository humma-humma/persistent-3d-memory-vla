import numpy as np

from persistent_scene_memory.robocasa_hierarchical import SKILLS
from persistent_scene_memory.robocasa_skill_merge import merge_skill_features


def _write(path, samples_per_skill):
    labels = np.repeat(np.arange(len(SKILLS)), samples_per_skill)
    count = len(labels)
    np.savez_compressed(
        path,
        features=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        labels=labels,
        frame_indices=np.arange(count),
        episode_roots=np.asarray([str(path)] * count),
        skills=np.asarray(SKILLS),
    )


def test_merge_caps_each_auxiliary_skill(tmp_path):
    primary, auxiliary, output = (tmp_path / name for name in ("p.npz", "a.npz", "o.npz"))
    _write(primary, 1)
    _write(auxiliary, 5)

    report = merge_skill_features(primary, auxiliary, output, auxiliary_per_skill=2)

    with np.load(output) as table:
        assert np.bincount(table["labels"]).tolist() == [3] * len(SKILLS)
    assert report["primary_samples"] == len(SKILLS)
    assert report["auxiliary_samples"] == 2 * len(SKILLS)
