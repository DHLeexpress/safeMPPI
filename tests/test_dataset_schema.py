import torch

from cfm_mppi.data.canonical_dataset import build_canonical_from_mizuta, save_canonical_splits


def test_mizuta_adapter_schema(tmp_path):
    raw = torch.randn(6, 9, 12)
    path = tmp_path / "train80_ego.pt"
    torch.save(raw, path)
    data = build_canonical_from_mizuta(path, history_len=4)
    assert data["states"].shape == (6, 13, 4)
    assert data["controls_si"].shape == (6, 12, 2)
    assert data["ego_history"].shape == (6, 4, 4)
    paths = save_canonical_splits(data, tmp_path / "canonical", seed=0)
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["test"].exists()
