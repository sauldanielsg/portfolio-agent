from typing import Literal
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

class Settings(BaseSettings):
    ai_enabled: bool = True
    ai_mode: Literal["mock", "gemini"] = "mock"
    
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    
    max_output_tokens: int = Field(default=500, ge=1, le=2000)
    daily_model_call_limit: int = Field(default=20, ge=0)
    quota_timezone: str = "America/Los_Angeles"
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
    )
    
    agent_backend: Literal["sdk", "langchain"] = "sdk"
    
    @model_validator(mode="after")
    def validate_configuration(self):
        try:
            ZoneInfo(self.quota_timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("quota_timezone must be a valid timezone") from error
        
        if self.ai_mode == "gemini":
            if (
                self.gemini_api_key is None
                or not self.gemini_api_key.get_secret_value().strip()
            ):
                raise ValueError("GEMINI_API_KEY is required in Gemini mode")
            if not self.gemini_model or not self.gemini_model.strip():
                raise ValueError("GEMINI_MODEL is required in Gemini mode")
        return self