import logging

from fastapi import APIRouter, Depends, HTTPException

from src.models.chat import ChatRequest, ChatResponse
from src.services.chat_orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

# APIRouter isola as rotas de chat do entrypoint principal (main.py),
# seguindo o princípio de Single Responsibility e facilitando testes unitários.
router = APIRouter(prefix="/api/v1", tags=["Chat"])


def get_orchestrator() -> ChatOrchestrator:
    """
    Dependência que será substituída no main.py pela instância singleton real.
    Declarada aqui para permitir override via app.dependency_overrides nos testes.
    """
    raise NotImplementedError("Orquestrador não configurado. Verifique o lifespan em main.py.")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Endpoint principal do assistente RAG",
    description=(
        "Recebe a mensagem atual do usuário e o histórico da conversa "
        "(compatível com o estado de sessão do Streamlit). "
        "Retorna a resposta gerada pelo LLaMA3 enriquecida com contexto "
        "recuperado semanticamente do banco vetorial Qdrant."
    ),
)
async def chat_endpoint(
    request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """
    Fluxo RAG completo:
        POST /api/v1/chat
        Body: { "message": str, "history": [ {role, content}, ... ] }
        → Embedding da mensagem → Busca Qdrant → LLaMA3 via Groq
        → { "reply": str, "sources": list[str] | null }
    """
    try:
        response = await orchestrator.process_message(request)
        return response

    except HTTPException:
        # Repassa HTTPExceptions já tipadas pelos serviços (429, 503, 504...)
        raise

    except ValueError as e:
        # Erros de validação de negócio (ex: mensagem mal formada)
        logger.warning(f"Erro de validação no endpoint /chat: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Erro inesperado no endpoint /chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao processar a requisição de chat. Tente novamente.",
        )
