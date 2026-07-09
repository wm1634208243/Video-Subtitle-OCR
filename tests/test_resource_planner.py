from app import resource_planner


def test_manual_workers_are_capped_by_max_workers(monkeypatch) -> None:
    monkeypatch.setenv("VSO_WORKERS", "4")
    monkeypatch.setenv("VSO_MAX_WORKERS", "2")
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 22)
    monkeypatch.setattr(resource_planner, "_memory_gb", lambda: (32.0, 24.0))

    plan = resource_planner.build_resource_plan()

    assert plan.worker_count == 2
    assert plan.worker_source == "manual"
    assert plan.max_workers == 2


def test_auto_workers_consider_cpu_and_memory(monkeypatch) -> None:
    monkeypatch.delenv("VSO_WORKERS", raising=False)
    monkeypatch.delenv("VSO_MAX_WORKERS", raising=False)
    monkeypatch.delenv("VSO_CPU_THREADS", raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 22)
    monkeypatch.setattr(resource_planner, "_memory_gb", lambda: (32.0, 24.0))

    plan = resource_planner.build_resource_plan()

    assert plan.worker_count == 2
    assert plan.worker_source == "auto"
    assert plan.cpu_threads == 11
