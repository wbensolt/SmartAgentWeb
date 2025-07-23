from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import AIMessage
from langchain.llms.base import LLM
from pydantic import PrivateAttr
import logging
from langchain_core.runnables import Runnable  # Assure-toi que ce chemin est correct

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain.llms import Ollama as OllamaLLM

from utils.config import get_config, reset_config


class LLMProvider(ABC):
    """Interface abstraite pour les providers LLM"""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


"""class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.2:latest", temperature: float = 0.7, **kwargs):
        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature, **kwargs)

    def invoke(self, prompt: str) -> str:
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Ollama",
            "model": self.model_name,
            "temperature": self.temperature
        }
"""

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", temperature: float = 0.7):#"meta-llama/llama-4-scout-17b-16e-instruct"llama3-70b-8192
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.llm = ChatGroq(model_name=model, api_key=api_key, temperature=temperature)

    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, AIMessage):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"❌ Erreur LLM: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Groq",
            "model": self.model_name,
            "temperature": self.temperature
        }


class GroqRunnableAdapter(Runnable):
    def __init__(self, groq_provider):
        self.groq_provider = groq_provider
        self.logger = logging.getLogger(__name__)

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        try:
            if hasattr(input, "to_string"):
                prompt = input.to_string()
            else:
                prompt = str(input)
            raw_response = self.groq_provider.invoke(prompt)
            return raw_response
        except Exception as e:
            self.logger.error(f"Erreur GroqRunnableAdapter: {str(e)}")
            return f"Erreur interne: {str(e)}"

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        return self.invoke(input, config)



class LLMManager:
    def __init__(self):
        reset_config()
        self.config = get_config()
        if self.config.llm_provider == "groq":
            self.provider = GroqProvider(
                api_key=self.config.groq_api_key,
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
        """else:
            self.provider = OllamaProvider(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )"""

    def get_llm(self) -> Runnable:
        # Si c’est GroqProvider, on renvoie l’adaptateur Runnable
        if isinstance(self.provider, GroqProvider):
            return GroqRunnableAdapter(self.provider)
        # OllamaProvider est supposé être compatible Runnable, sinon créer un adapter similaire
        return self.provider

    def set_provider(self, provider: LLMProvider):
        self.provider = provider

    def invoke(self, prompt: str) -> str:
        return self.provider.invoke(prompt)

    def get_model_info(self) -> Dict[str, Any]:
        return self.provider.get_model_info()


class LangChainLLMWrapper(LLM):
    _provider: LLMProvider = PrivateAttr()

    def __init__(self, provider: LLMProvider, **kwargs):
        super().__init__(**kwargs)
        self._provider = provider

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._provider.invoke(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"provider": self._provider.get_model_info()}

    @property
    def _llm_type(self) -> str:
        return "wrapped_llm"
