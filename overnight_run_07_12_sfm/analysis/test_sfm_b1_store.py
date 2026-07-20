import numpy as np

import sfm_b1_store as S


def result(y, *, resolved=True, full_h=True):
    if not resolved:
        return {"resolved": False, "error": "solver"}
    return dict(
        resolved=True, y=int(y), taskspace=True, collision_free=bool(y),
        certificate=bool(y), full_h=bool(full_h), terminal_step=10 if full_h else 3,
        train_eligible=bool(y and full_h), segment=np.zeros((11, 2), np.float32),
        pedestrian_prediction=np.zeros((11, 2, 2), np.float32), diagnostics={"slack": 0.1},
    )


def context(store, scenario=1, gamma=.1, step=0):
    return store.add_context(
        scenario_id=scenario, gamma=gamma, step=step, state=np.zeros(4),
        hp10=np.zeros((10, 16, 12)), low5=np.zeros(5), hist=np.zeros((16, 2)),
        ped_xy=np.zeros((2, 2)), ped_vel=np.zeros((2, 2)),
    )


def test_D_partitions_and_errors_enter_none(tmp_path):
    store = S.RoundShard(1)
    cid = context(store)
    store.add_resolved_query(cid, 0, np.zeros((10, 2)), .2, result(1), acquisition_step=0)
    store.add_resolved_query(cid, 1, np.zeros((10, 2)), .3, result(0), acquisition_step=1)
    store.add_resolved_query(cid, 2, np.zeros((10, 2)), .4, result(1, full_h=False), acquisition_step=2)
    store.add_error(context_key=(1, .1, 0), candidate_id=3, error="SOCP")
    assert len(store.D) == 3
    assert len(store.Dplus) == 1
    assert len(store.Dminus) == 1
    assert len(store.errors) == 1
    assert not store.queries[2]["train_eligible"]
    manifest = store.save(tmp_path / "round.pt")
    assert manifest["D"] == 3 and manifest["Dplus"] == 1 and manifest["Dminus"] == 1
    restored = S.RoundShard.load(tmp_path / "round.pt")
    assert restored.validate()["errors"] == 1
