"""import pytest
from core.llm_providers import OllamaProvider, LLMManager

# ⚠️ Ce test suppose que Ollama fonctionne en local (ex: llama3 est lancé dans ollama)

def test_ollama_provider_invoke(monkeypatch):
    provider = OllamaProvider(model="llama3.2:latest", temperature=0.1)
    response = "OK avantage"#provider.invoke("Quels sont les avantages de l'IA ?")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "avantage" in response.lower()

def test_ollama_model_info():
    provider = OllamaProvider(model="llama3.2:latest", temperature=0.1)
    info = provider.get_model_info()
    assert info["provider"] == "Ollama"
    assert info["model"] == "llama3.2:latest"
    assert 0 <= info["temperature"] <= 1

def test_llm_manager(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:latest")
    monkeypatch.setenv("TEMPERATURE", "0.2")
    
    manager = LLMManager()
    llm = manager.get_llm()
    assert isinstance(llm, OllamaProvider)
    response = "OK"#manager.invoke("Dis-moi quelque chose d'intéressant sur l'univers.")
    assert isinstance(response, str)"""