from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PREFIX_URL: str
    PIXABAY_API_KEY: str
    SERVER_RELOAD: bool

    PATH_BASE_MODEL_V10: str
    WEIGHT_PATH_BASE_MODEL_V10: str

    PATH_QUANTA_MODEL_V10: str
    WEIGHT_PATH_QUANTA_MODEL_V10: str

    PATH_BASE_MODEL_V11: str
    WEIGHT_PATH_BASE_MODEL_V11: str

    PATH_QUANTA_MODEL_V11: str
    WEIGHT_PATH_QUANTA_MODEL_V11: str

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8')


settings = Settings()
