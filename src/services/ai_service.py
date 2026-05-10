import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict

from groq import AsyncGroq
import groq
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from src.config.settings import settings

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Gera embeddings para o texto fornecido."""
        pass

    @abstractmethod
    async def generate_response(self, messages: List[Dict], system_prompt: str) -> str:
        """Gera uma resposta conversacional usando o modelo de linguagem."""
        pass


class GroqRAGService(BaseLLMService):
    def __init__(self):
        # A GROQ_API_KEY é instanciada isoladamente das configurações
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # SentenceTransformer é carregado UMA ÚNICA VEZ no __init__ (pesado, evita reload por request)
        self._encoder = SentenceTransformer(settings.embedding_model_name)

    async def get_embedding(self, text: str) -> list[float]:
        """
        Gera embeddings utilizando sentence-transformers com o modelo all-MiniLM-L6-v2.
        Dimensão de saída: 384 — valor vital para a configuração da coleção no Qdrant.

        ATENÇÃO: sentence-transformers é uma biblioteca SÍNCRONA e CPU-bound.
        Usamos asyncio.get_event_loop().run_in_executor() para evitar bloqueio
        do event loop do FastAPI durante a inferência local do modelo.
        """
        try:
            loop = asyncio.get_event_loop()
            # Executa em thread pool para não bloquear o event loop (operação CPU-bound/síncrona)
            embedding_array = await loop.run_in_executor(
                None, self._encoder.encode, text
            )
            return embedding_array.tolist()
        except Exception as e:
            logger.error(
                f"Erro na vetorização local (Sentence Transformers): {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail="Erro interno ao gerar embedding de texto."
            )

    async def generate_response(self, messages: List[Dict], system_prompt: str) -> str:
        """
        Gera resposta com llama3-8b-8192 via Groq, combinando
        system_prompt com o histórico completo do usuário.
        """
        try:
            # Prepara as mensagens combinando o system prompt com o histórico
            api_messages: List[Dict] = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                api_messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=api_messages,
            )
            return response.choices[0].message.content

        except groq.RateLimitError as e:
            logger.error(f"Groq API - Rate Limit excedido: {e}", exc_info=True)
            raise HTTPException(
                status_code=429,
                detail="Limite de requisições da Groq excedido. Tente novamente em instantes.",
            )

        except groq.APIConnectionError as e:
            logger.error(f"Groq API - Falha de conexão: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Serviço de IA temporariamente indisponível (falha de conexão).",
            )

        except groq.APITimeoutError as e:
            logger.error(f"Groq API - Timeout: {e}", exc_info=True)
            raise HTTPException(
                status_code=504,
                detail="Tempo limite excedido na resposta do provedor de IA.",
            )

        except Exception as e:
            logger.error(f"Erro inesperado no serviço Groq: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Erro interno durante a geração de resposta."
            )
