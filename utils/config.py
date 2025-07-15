# utils/config.py

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # charge les variables depuis .env

class SmartAgentConfig(BaseSettings):
    """Configuration centrale pour SmartAgentWeb"""

    # === Chemins ===
    data_path: Path = Field(default=Path("./data"), description="Dossier contenant les documents sources")
    vector_store_path: Path = Field(default=Path("./agents/vector"), description="Emplacement des index vectoriels")
    logs_path: Path = Field(default=Path("./logs"), description="Dossier pour logs")

    # === Modèles ===
    llm_provider: str = Field(default="groq", description="Provider de LLM (ollama ou groq)")
    llm_model: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", description="Nom du modèle LLM à utiliser")
    embedding_model: str = Field(default="paraphrase-multilingual:278m-mpnet-base-v2-fp16", description="Modèle pour les embeddings")

    # === Clés API ===
    #openai_api_key: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = Field(default=os.getenv("GROQ_API_KEY"))
    tavily_api_key: Optional[str] = Field(default=os.getenv("TAVILY_API_KEY"))

    # === Options LLM ===
    temperature: float = Field(default=0.7, description="Température du modèle LLM")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

# Fonction utilitaire pour récupérer la config (singleton)
_config: Optional[SmartAgentConfig] = None

def get_config() -> SmartAgentConfig:
    global _config
    if _config is None:
        _config = SmartAgentConfig()
    return _config

def reset_config():
    global _config
    _config = None
