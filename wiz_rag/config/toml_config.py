from pathlib import Path

import toml
from pydantic import BaseModel, Field

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 加载环境变量
try:
    with open(ROOT_DIR / "config.toml") as f:
        _toml_config = toml.load(f)
except FileNotFoundError:
    _toml_config = {}


# LLM 配置
class LLMConfig(BaseModel):
    # LLM 提供商
    provider: str = Field(
        default=_toml_config.get("llm", {}).get("provider", "deepseek"),
        description="LLM 提供商"
    )

    # LLM 模型名
    model_name: str = Field(
        default=_toml_config.get("llm", {}).get("model_name", "deepseek-chat"),
        description="LLM 模型名称"
    )

    # LLM API Key
    api_key: str = Field(
        default=_toml_config.get("llm", {}).get("api_key", ""),
        description="LLM API Key"
    )


# 应用配置
class TomlConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)


# 全局配置实例
toml_config = TomlConfig()
