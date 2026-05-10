import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.services.vector_store import VectorStoreService
from src.services.ai_service import BaseLLMService, GroqRAGService
from src.services.chat_orchestrator import ChatOrchestrator
from src.config.settings import settings
from src.routes import chat_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────
# Serviços pesados (SentenceTransformer, QdrantClient) são instanciados
# uma única vez no lifespan e reutilizados em todas as requisições.
# Isso elimina o overhead de recarregar o modelo por request.
_llm_service: BaseLLMService | None = None
_vector_store: VectorStoreService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação (substitui o depreciado @on_event).
    Inicializa os singletons e sobrescreve a dependência do router.
    """
    global _llm_service, _vector_store

    logger.info("🚀 Inicializando serviços RAG...")

    _llm_service = GroqRAGService()        # carrega SentenceTransformer (~90MB)
    _vector_store = VectorStoreService()   # conecta ao Qdrant

    try:
        _vector_store.ensure_collection(vector_size=settings.vector_dimension)
        logger.info("✅ Qdrant conectado e coleção verificada.")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant indisponível na inicialização: {e}")

    # ── Override da dependência do router ────────────────────────
    # Design Decision: o router declara get_orchestrator() como placeholder.
    # Aqui fazemos o override com a instância real que usa os singletons.
    # Isso mantém o router completamente testável via dependency_overrides.
    def _get_orchestrator_singleton() -> ChatOrchestrator:
        return ChatOrchestrator(_vector_store, _llm_service)

    app.dependency_overrides[chat_routes.get_orchestrator] = _get_orchestrator_singleton
    logger.info("✅ Serviços injetados com sucesso.")

    yield  # ← Aplicação roda aqui

    logger.info("🔻 Finalizando aplicação.")


# ── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="RAG Virtual Assistant API",
    description="Backend RAG: Groq (LLaMA3) + Sentence Transformers + Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────
# Permite requisições de qualquer origem durante o desenvolvimento.
# Em produção, substitua "*" pelo domínio real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna JSON padronizado."""
    logger.error(f"Exceção global não tratada: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": "Erro interno do servidor. Tente novamente mais tarde.",
            "detail": str(exc),
        },
    )


# ── Routers ───────────────────────────────────────────────────────
app.include_router(chat_routes.router)


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Verifica disponibilidade da API")
async def health_check():
    return {"status": "ok", "llm_model": settings.llm_model}
