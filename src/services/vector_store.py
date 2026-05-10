import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi import HTTPException

from src.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        # Design Decision: https=False e api_key=None desativam TLS explicitamente.
        # O Qdrant-client ≥1.11 ativa HTTPS automaticamente quando api_key não é None,
        # causando [SSL: WRONG_VERSION_NUMBER] em instâncias Docker sem certificado.
        # prefer_grpc=False garante transporte HTTP/REST puro (sem handshake gRPC/TLS).
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=None,        # Forçamos None → desativa modo seguro da lib
            https=False,         # Desativa TLS explicitamente para Docker local
            prefer_grpc=False,   # Usa REST puro; gRPC também tentaria TLS
        )
        self.collection_name = settings.vector_collection_name

    def ensure_collection(self, vector_size: int = settings.vector_dimension):
        """Cria a coleção no Qdrant se ainda não existir."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    f"Coleção '{self.collection_name}' criada com {vector_size} dimensões."
                )
        except Exception as e:
            logger.error(f"Falha ao criar/verificar coleção no Qdrant: {e}", exc_info=True)
            raise

    def search(self, query_vector: list[float], limit: int = 3) -> list[str]:
        """
        Executa busca semântica por similaridade no Qdrant.

        Usa query_points() — API moderna do qdrant-client >= 1.7.
        O método .search() foi removido nas versões recentes da biblioteca.

        Retorno: lista de strings com o texto de cada chunk recuperado,
        mapeado a partir do campo 'text' do payload de cada ScoredPoint.
        """
        try:
            # query_points() substitui o depreciado .search() no qdrant-client >= 1.7.
            # O vetor de consulta vai no parâmetro `query=`; os hits ficam em .points.
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            return [
                point.payload.get("text", "")
                for point in results.points
                if point.payload
            ]

        except UnexpectedResponse as e:
            logger.error(f"Qdrant retornou resposta inesperada: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Banco vetorial indisponível ou coleção não encontrada.",
            )
        except Exception as e:
            logger.error(f"Falha na busca semântica no Qdrant: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Erro interno na consulta ao banco vetorial.",
            )
