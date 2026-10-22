from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: Provider = "anthropic"
    # None uses the provider's default model (llm.DEFAULT_MODELS).
    model: str | None = None
    # None lets the SDK resolve credentials itself (ANTHROPIC_API_KEY, `ant auth login` profile...)
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    prompt_version: str = "v1"
    # DEBUG adds the prompts and raw response text to the log.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
