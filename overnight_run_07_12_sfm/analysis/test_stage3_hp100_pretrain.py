import json

import torch

import sfm_hp100_dynamics as DYN
import stage3_hp100_pretrain as P


def _write_one_gamma_dataset(root):
    episodes = torch.arange(500, dtype=torch.int64)
    # Give episode zero a second step so newest-to-oldest indexing is observable.
    episodes = torch.cat((torch.tensor([0]), episodes))
    steps = torch.cat((torch.tensor([0, 1]), torch.zeros(499, dtype=torch.int64)))
    hp = torch.zeros(len(episodes), 32, 100)
    hp[0].fill_(10.0)
    hp[1].fill_(11.0)
    payload = {
        "schema_version": P.SCHEMA_VERSION,
        "success_only": True,
        "gamma": 0.1,
        "n_traj": 500,
        "dynamics": DYN.contract(),
        "hp": hp,
        "low5": torch.zeros(len(episodes), 5),
        "hist": torch.zeros(len(episodes), 16, 2),
        "U": torch.zeros(len(episodes), 10, 2),
        "episode": episodes,
        "step": steps,
    }
    path = root / "sfm_hp100_windows_g0.1.pt"
    torch.save(payload, path)
    manifest = {
        "status": "HP100_ID_DATASET_COMPLETE",
        "schema_version": P.SCHEMA_VERSION,
        "files": [{
            "gamma": 0.1,
            "file": path.name,
            "sha256": P.sha256_file(path),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_loader_is_exact_500_lineage_disjoint_and_lazy_hp10(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    train, val, metadata = P.load_split(
        tmp_path, gammas=(0.1,), val_frac=0.1, seed=20260720
    )
    train_episodes = set(train.episodes.tolist())
    val_episodes = set(val.episodes.tolist())
    assert train_episodes.isdisjoint(val_episodes)
    assert len(train_episodes) == 450
    assert len(val_episodes) == 50
    assert metadata["files"]["0.1"]["train_lineages"] == 450
    assert metadata["files"]["0.1"]["val_lineages"] == 50

    source = train.sources[0]
    assert source["hp"].shape == (501, 32, 100)
    assert source["_history_indices"].shape == (501, 10)
    assert source["_history_indices"][1].tolist() == [1] + [0] * 9
    # The only ten-frame tensor is gathered for one requested row.
    dataset = train if 0 in train_episodes else val
    position = torch.nonzero(dataset.episodes == 0, as_tuple=False).flatten()[-1]
    hp10 = dataset[int(position)][0]
    assert hp10.shape == (10, 32, 100)
    assert not hasattr(dataset, "hp10")


def test_loader_rejects_less_than_500_successful_lineages(tmp_path):
    _write_one_gamma_dataset(tmp_path)
    path = tmp_path / "sfm_hp100_windows_g0.1.pt"
    payload = torch.load(path, weights_only=False)
    payload["n_traj"] = 499
    torch.save(payload, path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = P.sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    try:
        P.load_split(tmp_path, gammas=(0.1,))
    except ValueError as error:
        assert "exactly 500 successful lineages" in str(error)
    else:
        raise AssertionError("incomplete successful-lineage dataset was accepted")


def test_hierarchical_mass_is_gamma_trajectory_window():
    episodes = torch.tensor([1, 1, 2, 2, 2, 2, 7, 8, 8, 8])
    gammas = torch.tensor([0] * 6 + [1] * 4)
    weights = P.hierarchical_sampler_weights(episodes, gammas)
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert torch.isclose(weights[gammas == 0].sum(), torch.tensor(0.5, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 1].sum(), torch.tensor(0.25, dtype=weights.dtype))
    assert torch.isclose(weights[episodes == 2].sum(), torch.tensor(0.25, dtype=weights.dtype))


class _ValidationPolicy(torch.nn.Module):
    u_max = 2.0
    d = 20

    def ctx_from(self, hp, low, hist):
        return low

    def forward(self, x, tau, ctx):
        return 0.1 * x + 0.0 * ctx[:, :1]


def test_validation_cfm_is_fixed_and_preserves_rng():
    count = 6
    source = {
        "hp": torch.zeros(count, 32, 100),
        "low5": torch.zeros(count, 5),
        "hist": torch.zeros(count, 16, 2),
        "U": torch.zeros(count, 10, 2),
        "episode": torch.arange(count),
        "_history_indices": torch.arange(count)[:, None].expand(-1, 10).clone(),
    }
    dataset = P.HP100WindowDataset(
        [source, source],
        gamma_rows=torch.tensor([0, 0, 0, 1, 1, 1]),
        source_rows=torch.tensor([0, 1, 2, 3, 4, 5]),
    )
    torch.manual_seed(9)
    before = torch.random.get_rng_state().clone()
    first = P.deterministic_validation(
        _ValidationPolicy(), dataset, (0.1, 0.2), "cpu", batch=2, seed=17
    )
    after = torch.random.get_rng_state().clone()
    second = P.deterministic_validation(
        _ValidationPolicy(), dataset, (0.1, 0.2), "cpu", batch=2, seed=17
    )
    assert first == second
    assert torch.equal(before, after)


def test_promotion_gate_rejects_non_id_or_non_unit_temperature():
    good = {
        "distribution": "ID",
        "temperature": 1.0,
        "per_gamma": {"0.1": {"SR": 0.8, "CR": 0.2}},
    }
    P._validate_gate(good, (0.1,))
    for key, value in (("distribution", "OOD"), ("temperature", 0.5)):
        bad = dict(good)
        bad[key] = value
        try:
            P._validate_gate(bad, (0.1,))
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid promotion gate accepted: {key}={value}")


def test_warmup_cosine_does_not_zero_the_last_training_epoch():
    assert P._lr_multiplier(0, 120, 5) == 0.2
    assert P._lr_multiplier(4, 120, 5) == 1.0
    assert P._lr_multiplier(119, 120, 5) > 0.0
    assert P._lr_multiplier(120, 120, 5) == 0.0
