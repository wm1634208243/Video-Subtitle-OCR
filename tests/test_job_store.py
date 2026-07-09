from app.job_store import JobRecord, JobStore


def test_job_store_does_not_persist_secrets(tmp_path) -> None:
    store = JobStore()
    job = JobRecord(
        id="secret-test",
        filename="video.mp4",
        input_path=str(tmp_path / "video.mp4"),
        job_dir=str(tmp_path),
        options={},
        secrets={"llm_api_key": "secret-value"},
    )

    store.add(job)

    content = (tmp_path / "job.json").read_text(encoding="utf-8")
    assert "secret-value" not in content
    assert "secrets" not in content
