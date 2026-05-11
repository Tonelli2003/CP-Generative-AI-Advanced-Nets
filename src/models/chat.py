from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum


# ── Tipos Base ────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    """Modelo de mensagem individual usada no histórico estruturado."""
    role: MessageRole = Field(..., description="Role do remetente (user, assistant, system)")
    content: str = Field(..., min_length=1, description="Conteúdo da mensagem")

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


# ── Schema Compatível com Streamlit ───────────────────────────────
# Design Decision: o Streamlit envia a mensagem atual separada do histórico.
# Isso simplifica o front-end: não precisa montar o payload completo antes
# de enviar — o orquestrador cuida da composição final.

class ChatRequest(BaseModel):
    """
    Schema de requisição compatível com o front-end Streamlit.

    - message : mensagem atual do usuário (string)
    - history : histórico da conversa como lista de dicts {role, content}
                Streamlit mantém o histórico no estado da sessão e o envia
                a cada request, permitindo contexto multi-turn sem sessão server-side.
    """
    message: str = Field(..., min_length=1, description="Mensagem atual do usuário")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Histórico da conversa [ {role: str, content: str} ]"
    )

    @field_validator("history", mode="before")
    @classmethod
    def validate_history_roles(cls, v: list) -> list:
        """Garante que cada item do histórico tem os campos 'role' e 'content'."""
        valid_roles = {"user", "assistant", "system"}
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"history[{i}] deve ser um dicionário.")
            if "role" not in item or "content" not in item:
                raise ValueError(f"history[{i}] deve ter as chaves 'role' e 'content'.")
            if item["role"] not in valid_roles:
                raise ValueError(
                    f"history[{i}].role inválido: '{item['role']}'. "
                    f"Valores aceitos: {valid_roles}"
                )
        return v


class ChatResponse(BaseModel):
    """Schema de resposta retornado ao front-end Streamlit."""
    reply: str = Field(..., description="Resposta gerada pelo assistente (LLaMA3 via Groq)")
    sources: Optional[List[str]] = Field(
        None, description="Trechos recuperados do Qdrant usados como contexto RAG"
    )
