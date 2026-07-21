import sfm_b1_sweep as SW


def test_cpu_pools_keep_legacy_slots_and_add_four_disjoint_subslots(tmp_path):
    cpulist = tmp_path / "cpulist"
    cpulist.write_text("64-127\n")

    pools = SW.cpu_pools(cpulist)

    assert pools["1"] == list(range(64, 96))
    assert pools["3"] == list(range(96, 128))
    assert pools["1a"] == list(range(64, 80))
    assert pools["1b"] == list(range(80, 96))
    assert pools["3a"] == list(range(96, 112))
    assert pools["3b"] == list(range(112, 128))
    assert not (set(pools["1a"]) & set(pools["1b"]))
    assert not (set(pools["3a"]) & set(pools["3b"]))
    assert sorted(pools["1a"] + pools["1b"]) == pools["1"]
    assert sorted(pools["3a"] + pools["3b"]) == pools["3"]


def test_gpu_slot_mapping_reuses_the_physical_gpu_uuid():
    expected = {
        "1": "1", "1a": "1", "1b": "1",
        "3": "3", "3a": "3", "3b": "3",
    }
    for slot, gpu_index in expected.items():
        assert SW.gpu_index_for_slot(slot) == gpu_index
        assert SW.GPU_UUIDS[SW.gpu_index_for_slot(slot)] == SW.GPU_UUIDS[gpu_index]


def test_run_parallel_combines_subslot_cpu_pool_with_physical_gpu(monkeypatch, tmp_path):
    pools = {
        "1a": list(range(16)), "1b": list(range(16, 32)),
        "3a": list(range(32, 48)), "3b": list(range(48, 64)),
    }
    calls = []

    class CompletedProcess:
        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return CompletedProcess()

    monkeypatch.setattr(SW, "cpu_pools", lambda: pools)
    monkeypatch.setattr(SW, "_job_environment", lambda uuid: {"gpu_uuid": uuid})
    monkeypatch.setattr(SW.subprocess, "Popen", fake_popen)

    logs = SW.run_parallel([
        ("1a", ["python", "a.py"], "one_a"),
        ("1b", ["python", "b.py"], "one_b"),
        ("3a", ["python", "c.py"], "three_a"),
        ("3b", ["python", "d.py"], "three_b"),
    ], tmp_path)

    assert [call[0][2] for call in calls] == [
        SW._compact_cpu_list(pools[slot]) for slot in ("1a", "1b", "3a", "3b")
    ]
    assert [call[1]["gpu_uuid"] for call in calls] == [
        SW.GPU_UUIDS["1"], SW.GPU_UUIDS["1"],
        SW.GPU_UUIDS["3"], SW.GPU_UUIDS["3"],
    ]
    assert logs == [
        str(tmp_path / "one_a.log"), str(tmp_path / "one_b.log"),
        str(tmp_path / "three_a.log"), str(tmp_path / "three_b.log"),
    ]


def test_unknown_gpu_slot_is_rejected():
    try:
        SW.gpu_index_for_slot("2a")
    except ValueError as error:
        assert "unknown GPU/CPU slot" in str(error)
    else:
        raise AssertionError("unknown slot was accepted")
