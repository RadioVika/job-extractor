from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "llama"]
# Reasoning levels Bonsai 2's chat template accepts ("high" returns HTTP 500).
LlamaReasoning = Literal["none", "low", "medium", "xhigh"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: Provider = "anthropic"
    # None uses the provider's default model (llm.DEFAULT_MODELS).
    model: str | None = None
    # None lets the SDK resolve credentials itself (ANTHROPIC_API_KEY, `ant auth login` profile...)
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Local model: extract.main starts llama-server with this GGUF for the run.
    llama_server_bin: str = "llama-server"
    llama_model_path: Path | None = None
    llama_port: int = 8080
    # Room for the system prompt, the posting and MAX_TOKENS of output including reasoning.
    # PrismML suggests 65536 for Bonsai 2, but with the 27B model that runs out of GPU memory
    # on a 16 GB Mac ("Compute error." on every request).
    llama_ctx_size: int = 32768
    # None follows EFFORT. "none" skips reasoning, which is most of the local model's run time.
    llama_reasoning: LlamaReasoning | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    prompt_version: str = "v1"
    # DEBUG adds the prompts and raw response text to the log.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
