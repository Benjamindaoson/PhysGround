import json
from pathlib import Path
import tempfile

import numpy as np

from physground.contracts import PhysicsParams, Pose2D, PushAction
from physground.dataset import generate_transition_dataset, sample_inputs
from physground.dynamics import PlanarPushModel


def test_reduced_dataset_is_sharded_and_resumable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "data"
        first = generate_transition_dataset(
            output_dir=root,
            records=257,
            shard_size=128,
            seed=7,
            backend="reduced",
            resume=True,
        )
        assert first["records"] == 257
        assert [item["records"] for item in first["shards"]] == [128, 128, 1]
        hashes = [item["sha256"] for item in first["shards"]]

        second = generate_transition_dataset(
            output_dir=root,
            records=257,
            shard_size=128,
            seed=7,
            backend="reduced",
            resume=True,
        )
        assert [item["sha256"] for item in second["shards"]] == hashes
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["backend"] == "reduced"
        assert manifest["records"] == 257


def test_vectorized_reduced_generator_matches_reference_model() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "data"
        generate_transition_dataset(
            output_dir=root,
            records=8,
            shard_size=8,
            seed=13,
            backend="reduced",
        )
        with np.load(root / "part-00000.npz") as loaded:
            before = loaded["before"]
            action = loaded["action"]
            physics = loaded["physics"]
            after = loaded["after"]

        model = PlanarPushModel()
        for index in range(8):
            expected = model.step(
                Pose2D(*[float(value) for value in before[index]]),
                PushAction(*[float(value) for value in action[index]]),
                PhysicsParams.from_array(physics[index]),
            ).as_array()
            assert np.allclose(after[index], expected, atol=1e-6)


def test_sample_inputs_respect_contract_ranges() -> None:
    before, action, physics = sample_inputs(100, seed=1)
    assert before.shape == (100, 3)
    assert action.shape == (100, 5)
    assert physics.shape == (100, 4)
    assert np.all((physics[:, 0] >= 0.1) & (physics[:, 0] <= 1.2))
    assert np.all((action[:, 3] >= 0.005) & (action[:, 3] <= 0.08))
