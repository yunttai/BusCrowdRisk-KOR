from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    public_data_service_key: str
    its_api_key: str
    data_gg_service_key: str
    db_path: Path
    timeout_seconds: int = 10

    @property
    def has_service_key(self) -> bool:
        return bool(self.public_data_service_key)


def load_settings() -> Settings:
    load_dotenv()
    timeout = int(os.getenv("PUBLIC_DATA_TIMEOUT_SECONDS", "10"))
    public_data_service_key = os.getenv("PUBLIC_DATA_SERVICE_KEY", "")
    return Settings(
        public_data_service_key=public_data_service_key,
        its_api_key=os.getenv("ITS_API_KEY", ""),
        data_gg_service_key=os.getenv("DATA_GG_SERVICE_KEY", ""),
        db_path=Path(os.getenv("BUSSEAT_DB_PATH", "data/busseat.db")),
        timeout_seconds=timeout,
    )


def require_named_key(value: str, env_name: str, description: str) -> str:
    if not value:
        raise RuntimeError(f"{env_name}가 비어 있습니다. .env에 {description} 키를 넣으세요.")
    return value


def require_service_key(settings: Settings) -> str:
    return require_named_key(settings.public_data_service_key, "PUBLIC_DATA_SERVICE_KEY", "공공데이터포털 기본")
