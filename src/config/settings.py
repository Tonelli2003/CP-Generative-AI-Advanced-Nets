from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Configurações da Groq
    groq_api_key: str = Field(..., alias="GROQ_API_KEY", description="Chave de API obrigatória da Groq")
    
    # Configurações do Qdrant
    qdrant_host: str = Field(..., alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY")
    
    # Parâmetros de IA
    llm_model: str = Field("llama3-8b-8192", alias="LLM_MODEL")
    embedding_model_name: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
    
    # Configurações de Vetores
    vector_collection_name: str = Field("rag_collection", alias="VECTOR_COLLECTION_NAME")
    vector_dimension: int = Field(384, alias="VECTOR_DIMENSION")

    # Validador para garantir integridade do ambiente antes da inicialização do back-end
    @field_validator("groq_api_key", mode="before")
    def validate_groq_key(cls, v):
        if not v or v == "gsk_your_groq_api_key_here" or v.strip() == "":
            raise ValueError(
                "A variável GROQ_API_KEY não foi configurada corretamente. "
                "O back-end não pode ser iniciado sem credenciais válidas da IA."
            )
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
