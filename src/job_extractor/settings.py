from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # None lets the SDK resolve credentials itself (ANTHROPIC_API_KEY, `ant auth login` profile...)
    anthropic_api_key: SecretStr | None = None
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    prompt_version: str = "v1"
    # DEBUG adds the prompts and raw response text to the log.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
