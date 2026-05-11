<div align="center">

# 🤖 Assistente Virtual RAG

### CP · Generative AI Advanced Net · FIAP

<p>
  <img src="https://img.shields.io/badge/Python-3.12%2B%20%7C%203.14-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Groq-LLaMA_3-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
  <img src="https://img.shields.io/badge/Qdrant-Banco_Vetorial-DC244C?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

</div>

---

## 📌 Sobre o Projeto

Este projeto implementa um **Assistente Virtual inteligente** baseado no padrão arquitetural **RAG (Retrieval Augmented Generation)** para a disciplina de Generative AI Advanced Net da FIAP.

O sistema recebe perguntas em linguagem natural, busca contexto relevante em uma **base de conhecimento vetorizada** (Qdrant) e gera respostas precisas utilizando o modelo **LLaMA 3 via Groq Cloud API**, tudo orquestrado por uma API REST em **FastAPI**.

### Funcionalidades Principais

| Funcionalidade | Descrição |
| :--- | :--- |
| **Busca Semântica** | Recuperação de contexto via embeddings locais (all-MiniLM-L6-v2) |
| **LLM Conversacional** | Respostas geradas pelo LLaMA 3 (llama3-8b-8192) via Groq |
| **Histórico de Conversa** | Suporte a múltiplos turnos de conversa com contexto acumulado |
| **Injeção de Dependência** | Arquitetura desacoplada com padrão Abstract Base Class + FastAPI Depends |
| **Tratamento de Erros** | Handler global + logging estruturado por camada de serviço |

---

## 👥 Equipe

| Nome | RM |
| :--- | :---: |
| Augusto Oliveira Codo de Sousa | RM562080 |
| Felipe de Oliveira Cabral | RM561720 |
| Gabriel Tonelli Avelino Dos Santos | RM564705 |
| Vinícius Adrian Siqueira de Oliveira | RM564962 |
| Sofia Bueris Netto de Souza | RM565818 |

> **Instituição:** FIAP · **Disciplina:** Generative AI Advanced Net · **Turma:** 2TIAPF 2026

---

## 🏗️ Arquitetura RAG

```
Usuário (Front-end / Cliente HTTP)
        │
        │  POST /api/v1/chat  { messages: [...] }
        ▼
  ┌─────────────────────────────┐
  │       FastAPI               │  ← Roteamento + Validação (Pydantic)
  │   (chat_endpoint)           │
  └─────────┬───────────────────┘
            │
            ▼
  ┌─────────────────────────────┐
  │    ChatOrchestrator         │  ← Orquestração do Fluxo RAG
  └──────┬──────────────────────┘
         │
   ┌─────┴──────────────────────────────────┐
   │                                        │
   ▼                                        ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  GroqRAGService       │     │   VectorStoreService      │
│                      │     │                          │
│  get_embedding()     │     │  search(vector)           │
│  → SentenceTransformer│    │  → Qdrant (cosine sim.)  │
│    (all-MiniLM-L6-v2)│     │                          │
│    Dim: 384           │     │  ensure_collection()     │
│                      │     │  → Cria coleção na init  │
│  generate_response() │     └──────────────────────────┘
│  → Groq API          │              │
│    (llama3-8b-8192)  │              │ contexto recuperado
└──────────────────────┘              │
         │                            │
         └────────────────────────────┘
                      │
                      ▼
           Resposta enriquecida
           { reply: "...", sources: [...] }
```

---

## 📂 Estrutura do Projeto

```text
├── backend/
│   ├── src/                  # Código-fonte do FastAPI
│   ├── data/                 # Documentos brutos (PDFs)
│   ├── scripts/              # Scripts de ingestão
│   ├── docker-compose.yml    # Orquestração do Qdrant
│   ├── .env.example
│   └── requirements.txt      # Dependências do backend (e frontend)
├── frontend/
│   └── app.py                # Interface Streamlit
├── .venv/                    # Ambiente virtual (na raiz)
├── .gitignore
└── README.md
```

---

## 🛠️ Stack Tecnológica

| Categoria | Tecnologia |
| :--- | :--- |
| **Linguagem** | Python 3.10+ |
| **Framework Web** | [FastAPI](https://fastapi.tiangolo.com/) 0.111 |
| **LLM (Cloud)** | [Groq API](https://console.groq.com/) — Llama 3 (llama3-8b-8192) |
| **Embeddings (Local)** | [Sentence Transformers](https://www.sbert.net/) — all-MiniLM-L6-v2 (384 dim) |
| **Banco Vetorial** | [Qdrant](https://qdrant.tech/) via Docker |
| **Configurações** | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| **Servidor ASGI** | Uvicorn |

---

## 🚀 Como Executar

### Pré-requisitos

- Python **3.12+** instalado (testado até 3.14.3)
- [Docker](https://www.docker.com/) instalado e em execução
- Conta e API Key na [Groq Cloud](https://console.groq.com/) (gratuito)

---

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

### 2. Configure as variáveis de ambiente

Copie o template e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env` com sua chave real:

```env
GROQ_API_KEY=gsk_SUA_CHAVE_REAL_AQUI
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
LLM_MODEL=llama3-8b-8192
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
VECTOR_COLLECTION_NAME=rag_collection
VECTOR_DIMENSION=384
```

> ⚠️ **Importante:** deixe `QDRANT_API_KEY` **em branco** para execuções locais via Docker. Atribuir um valor a essa variável força o cliente a usar TLS/SSL, causando erro de conexão com a instância local sem certificado.

### 3. Suba o banco vetorial com Docker

```bash
docker compose up -d
```

> ✅ O Qdrant ficará disponível em `http://localhost:6333`. O painel web estará em `http://localhost:6333/dashboard`.

### 4. Crie o ambiente virtual e instale as dependências

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> ⚠️ O modelo `all-MiniLM-L6-v2` (~90MB) será baixado automaticamente pelo Sentence Transformers na primeira execução.

### 5. Execute o Back-end e o Front-end

O projeto requer **dois terminais abertos simultaneamente**. Como organizamos o projeto em pastas, você precisa entrar na pasta respectiva em cada terminal.

**Terminal 1 — Back-end (FastAPI)**

```bash
cd backend
python -m uvicorn src.main:app --reload
```

> 💡 O prefixo `python -m` invoca o uvicorn como módulo Python, contornando a política de Controle de Aplicativo do Windows que bloqueia executáveis `.exe` não assinados diretamente.

A API estará disponível em: **`http://localhost:8000`**  
Documentação interativa (Swagger UI): **`http://localhost:8000/docs`**

---

**Terminal 2 — Front-end (Streamlit)**

```bash
cd frontend
streamlit run app.py
```

A interface estará disponível em: **`http://localhost:8501`**

---

## 📡 Endpoints da API

| Método | Rota | Descrição |
| :---: | :--- | :--- |
| `GET` | `/health` | Health check — verifica se a API está ativa |
| `POST` | `/api/v1/chat` | Endpoint principal do assistente RAG |

### Exemplo de Requisição

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "O que é RAG?" }
    ]
  }'
```

### Exemplo de Resposta

```json
{
  "reply": "RAG (Retrieval Augmented Generation) é uma técnica que combina...",
  "sources": [
    "RAG é um padrão arquitetural que enriquece o contexto do LLM com...",
    "A busca semântica via embeddings permite recuperar documentos relevantes..."
  ]
}
```

---

## 🚀 Testando o Sistema

Com ambos os serviços rodando, acesse **`http://localhost:8501`** e utilize as perguntas abaixo para validar o fluxo RAG ponta a ponta:

| Categoria | Pergunta sugerida |
| :--- | :--- |
| **Extração de Dados** | *"Quais são os limites toleráveis de temperatura e vibração para os motores?"* |
| **Arquitetura** | *"Quais técnicas de Prompt Engineering foram implementadas neste sistema?"* |
| **Raciocínio Preditivo** | *"Um motor operando a 90°C e 14Hz exige qual ação de manutenção?"* |
| **Privacidade / Design** | *"Por que o projeto prioriza o uso de modelos locais e inferência offline?"* |

> As respostas devem ser geradas **com base exclusiva nos documentos indexados** no Qdrant. Ausência de contexto retorna a mensagem padrão de fallback.

---

## 🔐 Segurança

- O arquivo `.env` está listado no `.gitignore` e **nunca** deve ser versionado
- Use apenas o `.env.example` (com valores fictícios) no repositório
- A `GROQ_API_KEY` é validada via `pydantic-settings` na inicialização — a aplicação **não sobe** se a chave estiver ausente ou com valor padrão
- `QDRANT_API_KEY` deve ser **deixada em branco** no `.env` local — um valor preenchido habilita TLS e causa erro de SSL na conexão com o Docker sem certificado

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos — **FIAP · Generative AI Advanced Net · 2026**.
