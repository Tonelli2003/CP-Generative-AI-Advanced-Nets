"""
Pipeline de Ingestão de Documentos — RAG Backend
=================================================
Lê arquivos .pdf de data/raw_docs/, aplica chunking com overlap
e indexa os vetores no Qdrant via sentence-transformers local.

Uso:
    python scripts/index_documents.py
"""

import os
import sys
import uuid
import hashlib
import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# Garante que imports de src/ funcionem ao rodar da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Variáveis de Ambiente ─────────────────────────────────────────
load_dotenv()

QDRANT_HOST: str      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int      = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str  = os.getenv("VECTOR_COLLECTION_NAME", "rag_collection")
VECTOR_DIM: int       = int(os.getenv("VECTOR_DIMENSION", "384"))
EMBEDDING_MODEL: str  = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

RAW_DOCS_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_docs"

# ── Parâmetros de Chunking ────────────────────────────────────────
# Justificativa: chunk de 1000 chars captura parágrafos completos;
# overlap de 200 preserva contexto semântico entre chunks adjacentes.
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 64


# ═══════════════════════════════════════════════════════════════════
# LEITURA DE PDF
# ═══════════════════════════════════════════════════════════════════

def read_pdf_with_pypdf2(file_path: Path) -> str:
    """
    Extrai texto de um PDF usando PyPDF2.
    Adequado para PDFs simples com texto estruturado.
    """
    import PyPDF2

    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            logger.debug(f"  [PyPDF2] Página {i + 1}/{total} lida.")
    return "\n".join(text_parts)


def read_pdf_with_pdfplumber(file_path: Path) -> str:
    """
    Extrai texto de um PDF usando pdfplumber.
    Fallback robusto: lida melhor com colunas, tabelas e
    caracteres especiais (acentos, símbolos técnicos).
    """
    import pdfplumber

    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            logger.debug(f"  [pdfplumber] Página {i + 1}/{total} lida.")
    return "\n".join(text_parts)


def read_pdf(file_path: Path) -> str:
    """
    Tenta extrair texto com PyPDF2. Se falhar ou retornar vazio,
    usa pdfplumber como fallback automático.
    """
    try:
        text = read_pdf_with_pypdf2(file_path)
        if text.strip():
            logger.info(f"  → Extraído via PyPDF2: {len(text):,} caracteres.")
            return text
        logger.warning("  → PyPDF2 retornou texto vazio. Tentando pdfplumber...")
    except Exception as e:
        logger.warning(f"  → PyPDF2 falhou ({e}). Tentando pdfplumber...")

    try:
        text = read_pdf_with_pdfplumber(file_path)
        logger.info(f"  → Extraído via pdfplumber: {len(text):,} caracteres.")
        return text
    except Exception as e:
        logger.error(f"  → pdfplumber também falhou: {e}")
        return ""


def read_txt(file_path: Path) -> str:
    """Lê arquivo de texto simples com detecção de encoding."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"  → Não foi possível determinar o encoding de '{file_path.name}'.")
    return ""


# ═══════════════════════════════════════════════════════════════════
# CHUNKING — Sliding Window com Overlap
# ═══════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide o texto em chunks usando a estratégia de Sliding Window.

    Parâmetros:
        chunk_size : tamanho máximo de cada chunk em caracteres (padrão: 1000)
        overlap    : sobreposição entre chunks consecutivos em caracteres (padrão: 200)

    Design Decision:
        A sobreposição garante que contexto semântico não seja perdido
        nas fronteiras entre chunks. Isso melhora diretamente a qualidade
        da recuperação semântica no Qdrant (Requisito de Design — RAG).
    """
    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # avança respeitando o overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════
# QDRANT — Coleção e Indexação
# ═══════════════════════════════════════════════════════════════════

def ensure_collection(client: QdrantClient) -> None:
    """Verifica se a coleção existe no Qdrant. Se não, cria."""
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                f"✅ Coleção '{COLLECTION_NAME}' criada "
                f"(dimensão={VECTOR_DIM}, distância=Cosine)."
            )
        else:
            logger.info(f"ℹ️  Coleção '{COLLECTION_NAME}' já existe.")
    except UnexpectedResponse as e:
        logger.error(f"Erro ao verificar/criar coleção no Qdrant: {e}")
        raise


def index_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[str],
    source_filename: str,
) -> int:
    """
    Gera embeddings e envia os vetores ao Qdrant em lotes (batches).

    Payload de cada ponto:
        {
            "text":   <texto original do chunk>,
            "source": <nome do arquivo de origem>
        }

    Retorna o número de pontos indexados com sucesso.
    """
    total_indexed = 0

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]

        # Geração de embeddings em lote (mais eficiente que chunk-a-chunk)
        vectors = model.encode(batch, show_progress_bar=False).tolist()

        points = [
            models.PointStruct(
                # ID determinístico: mesmo arquivo + mesmo índice → mesmo UUID.
                # Garante idempotência: re-executar o script faz upsert no mesmo
                # ponto, nunca criando duplicatas na coleção do Qdrant.
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{source_filename}::{batch_start + i}",
                )),
                vector=vector,
                payload={"text": chunk, "source": source_filename},
            )
            for i, (chunk, vector) in enumerate(zip(batch, vectors))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_indexed += len(points)

        logger.info(
            f"  ↳ Lote {batch_start // BATCH_SIZE + 1}: "
            f"{len(points)} chunks indexados "
            f"({batch_start + len(points)}/{len(chunks)} total)."
        )

    return total_indexed


# ═══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def run_ingestion() -> None:
    logger.info("=" * 60)
    logger.info("🚀 Iniciando pipeline de ingestão de documentos RAG")
    logger.info("=" * 60)

    # 1. Valida a pasta de documentos
    if not RAW_DOCS_PATH.exists():
        RAW_DOCS_PATH.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Pasta '{RAW_DOCS_PATH}' foi criada agora. "
            "Adicione arquivos .txt ou .pdf e execute o script novamente."
        )
        return

    doc_files = sorted(
        f for f in RAW_DOCS_PATH.iterdir()
        if f.suffix.lower() in {".txt", ".pdf"}
    )

    if not doc_files:
        logger.warning(f"Nenhum arquivo .txt ou .pdf encontrado em '{RAW_DOCS_PATH}'.")
        return

    logger.info(f"📂 {len(doc_files)} arquivo(s) encontrado(s) para processar.")

    # 2. Conecta ao Qdrant
    logger.info(f"🔗 Conectando ao Qdrant → {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 3. Garante que a coleção existe
    ensure_collection(client)

    # 4. Carrega o modelo de embeddings (operação pesada, feita uma única vez)
    logger.info(f"🤖 Carregando modelo de embeddings: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Modelo de embeddings pronto.")

    # 5. Processa cada documento
    total_indexed_global = 0

    for doc_file in doc_files:
        logger.info(f"\n📄 Lendo arquivo: '{doc_file.name}'")

        if doc_file.suffix.lower() == ".pdf":
            raw_text = read_pdf(doc_file)
        else:
            raw_text = read_txt(doc_file)

        if not raw_text.strip():
            logger.warning(f"  ⚠️  '{doc_file.name}' sem texto extraível. Pulando.")
            continue

        # 6. Aplica chunking com overlap
        chunks = chunk_text(raw_text)
        logger.info(
            f"  → {len(chunks)} chunk(s) gerado(s) "
            f"(tamanho={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} chars)."
        )

        # 7. Indexa no Qdrant
        indexed = index_chunks(client, model, chunks, source_filename=doc_file.name)
        total_indexed_global += indexed
        logger.info(f"  ✅ '{doc_file.name}' → {indexed} chunks indexados.")

    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 Ingestão concluída! Total indexado: {total_indexed_global} chunk(s)")
    logger.info(f"   Coleção : '{COLLECTION_NAME}'")
    logger.info(f"   Qdrant  : {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
