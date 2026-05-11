import logging
from typing import List, Dict

from src.services.vector_store import VectorStoreService
from src.services.ai_service import BaseLLMService
from src.models.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orquestrador central do fluxo RAG (Retrieval Augmented Generation).

    Recebe instâncias de VectorStoreService e BaseLLMService via
    Injeção de Dependência, garantindo baixo acoplamento e testabilidade.
    """

    def __init__(self, vector_store: VectorStoreService, llm_service: BaseLLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Pipeline RAG completo:
            1. Vetoriza a mensagem atual do usuário (Sentence Transformers)
            2. Recupera chunks relevantes do Qdrant (busca semântica)
            3. Monta system_prompt rigoroso ancorado no contexto recuperado
            4. Compõe o histórico e envia ao LLM (Groq / LLaMA3)
            5. Retorna a resposta com as fontes usadas
        """
        user_message = request.message.strip()

        if not user_message:
            logger.warning("Mensagem vazia recebida pelo orquestrador.")
            return ChatResponse(reply="Por favor, envie uma mensagem para continuar.")

        logger.info(f"Processando mensagem: '{user_message[:80]}...'")

        # ── Passo 1: Gera embedding para a query atual ────────────
        # Usa apenas a mensagem atual (não o histórico) para o embedding,
        # pois a busca semântica deve recuperar contexto relevante ao
        # que o usuário perguntou AGORA, não à conversa inteira.
        query_vector = await self.llm_service.get_embedding(user_message)

        # ── Passo 2: Busca semântica no Qdrant ────────────────────
        context_docs = self.vector_store.search(query_vector, limit=3)
        context_str = "\n\n---\n\n".join(context_docs) if context_docs else ""

        if context_docs:
            logger.info(f"{len(context_docs)} chunk(s) recuperado(s) do Qdrant.")
        else:
            logger.warning("Nenhum contexto encontrado no Qdrant para esta query.")

        # ── Passo 3: System prompt rigoroso com contexto RAG ──────
        # Design Decision: instruir o LLM a se limitar ao contexto recuperado
        # evita alucinações e garante que as respostas sejam rastreáveis
        # aos documentos indexados, atendendo ao requisito de confiabilidade do RAG.
        if context_str:
            system_prompt = (
                "Você é um assistente virtual especializado e preciso. "
                "Sua única fonte de verdade é o CONTEXTO fornecido abaixo. "
                "Responda à pergunta do usuário de forma direta e objetiva, "
                "baseando-se EXCLUSIVAMENTE nas informações do contexto. "
                "Se a resposta não puder ser encontrada no contexto, diga: "
                "'Não encontrei informações sobre isso na base de conhecimento.' "
                "Nunca invente informações ou use conhecimento externo ao contexto.\n\n"
                f"=== CONTEXTO ===\n{context_str}\n================"
            )
        else:
            # Fallback sem contexto: avisa o usuário, mas não bloqueia a resposta
            system_prompt = (
                "Você é um assistente virtual. "
                "Não foram encontrados documentos relevantes na base de conhecimento "
                "para a pergunta atual. Informe isso ao usuário de forma gentil "
                "e sugira que ele reformule a pergunta ou consulte outras fontes."
            )

        # ── Passo 4: Compõe histórico no formato da API da Groq ───
        # O histórico vem do Streamlit como list[dict], já no formato correto.
        # Adicionamos a mensagem atual ao final para representar o turno presente.
        conversation: List[Dict[str, str]] = list(request.history)
        conversation.append({"role": "user", "content": user_message})

        # ── Passo 5: Gera resposta via Groq ──────────────────────
        reply = await self.llm_service.generate_response(conversation, system_prompt)
        logger.info("Resposta gerada com sucesso.")

        return ChatResponse(reply=reply, sources=context_docs if context_docs else None)
