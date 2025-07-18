from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Mapping, Optional
from pydantic import PrivateAttr
from core.llm_providers import LLMProvider


class LangChainChatWrapper(BaseChatModel):
    _provider: LLMProvider = PrivateAttr()

    def __init__(self, provider: LLMProvider, **kwargs):
        super().__init__(**kwargs)
        self._provider = provider

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        functions: Optional[Any] = None,  # <-- ajouter ce paramètre
        **kwargs
    ) -> ChatResult:
        prompt = "\n".join([m.content for m in messages if hasattr(m, "content")])
        response = self._provider.invoke(prompt)
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content=response))
        ])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._provider.get_model_info()

    @property
    def _llm_type(self) -> str:
        return "wrapped_chat_llm"
