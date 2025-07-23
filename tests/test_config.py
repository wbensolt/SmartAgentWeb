"""import os
import pytest
from pathlib import Path
from utils.config import get_config

def test_get_config_loads_env_variables(monkeypatch):
    # On simule des variables d'environnement
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("TEMPERATURE", "0.9")

    config = get_config()

    assert config.llm_provider == "groq"
    assert config.groq_api_key == "test_groq_key"
    assert config.temperature == 0.9
    assert isinstance(config.vector_store_path, Path)
"""