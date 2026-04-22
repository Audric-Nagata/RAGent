"""Application configuration — loads env vars with strict validation."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

import os
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Strict environment configuration with defaults where sensible."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM API Keys ──────────────────────────────────────────
    hf_token: str = os.getenv("HF_TOKEN", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # ── Observability ─────────────────────────────────────────────
    logfire_token: str = os.getenv("LOGFIRE_TOKEN")

    # ── Qdrant Vector DB ─────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "ragent-chunks"

    # ── Embedding ──────────────────────────────────────────────────
    embedding_dimension: int = 1024  # Qwen/Qwen3-Embedding-0.6B

    # ── Tavily Web Search ─────────────────────────────────────────
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")
