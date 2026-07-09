from app import user_config


def test_save_llm_config_does_not_return_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(user_config, "CONFIG_PATH", tmp_path / "config.json")

    public = user_config.save_llm_config(
        {
            "enabled": True,
            "provider": "openai-compatible",
            "model": "gpt-test",
            "base_url": "http://localhost/v1",
            "api_key": "secret",
        }
    )

    assert public["has_api_key"] is True
    assert "api_key" not in public
    assert user_config.get_llm_config(include_secret=True)["api_key"] == "secret"


def test_save_llm_config_keeps_existing_secret_when_blank(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(user_config, "CONFIG_PATH", tmp_path / "config.json")
    user_config.save_llm_config({"enabled": True, "provider": "openai", "api_key": "secret"})

    user_config.save_llm_config({"enabled": False, "provider": "ollama", "api_key": ""})

    config = user_config.get_llm_config(include_secret=True)
    assert config["provider"] == "ollama"
    assert config["api_key"] == "secret"


def test_reset_llm_config_removes_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(user_config, "CONFIG_PATH", tmp_path / "config.json")
    user_config.save_llm_config({"enabled": True, "provider": "openai", "api_key": "secret"})

    public = user_config.reset_llm_config()

    assert public["has_api_key"] is False
    assert not (tmp_path / "config.json").exists()
