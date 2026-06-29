"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OnCall Mind"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空，实际从.env加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"

    # LLM 统一调用配置（P0-1.1: 收敛所有硬编码参数）
    llm_temperature: float = 0.0        # 默认温度（规划/执行类节点使用 0，对话类可覆盖）
    llm_chat_temperature: float = 0.7   # 对话 Agent 专用温度
    llm_timeout: int = 60               # API 调用超时（秒）
    llm_max_retries: int = 2            # 最大重试次数
    llm_fallback_model: str = "qwen-plus"  # Fallback 模型（P0-1.2），为空则不启用

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒
    milvus_nprobe: int = 32     # 搜索 nprobe（P0-1.3: 10→32 提升召回率）

    # ------------------------------------------------------------------
    # RAG 检索配置（三层语义）
    #
    #   coarse_top_k  → 混合检索粗排候选数，供精排器筛选
    #   final_top_k   → 精排后最终返回给 LLM 的文档数
    #   eval_top_k    → 评估脚本专用，独立于线上运行配置
    #
    # 参数分工：
    #   - rag_top_k           basic 模式最终返回数
    #   - reranker_top_k      enhanced 模式最终返回数（精排后截断至此数量）
    #   - rerank_coarse_top_k enhanced 模式粗排候选数（混合检索召回数）
    #   - rag_mode            检索模式切换，basic 和 enhanced 各有独立 top_k
    # ------------------------------------------------------------------
    rag_top_k: int = 3
    rag_model: str = "qwen-max"
    rag_mode: Literal["basic", "enhanced"] = "basic"

    # Enhanced RAG 配置（rag_mode="enhanced" 时生效）
    query_preprocessor_type: Literal["none", "rewrite"] = "none"
    reranker_type: Literal["none", "cross_encoder"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_path: str = ""    # 本地模型路径，非空时优先从本地加载（跳过 HF 下载）
    reranker_top_k: int = 3          # enhanced 模式最终返回数（精排后截断）
    rerank_coarse_top_k: int = 10    # enhanced 模式粗排候选数（P0-1.3: 20→10 精排耗时减半）
    rag_diversify_by_file: bool = False  # 是否在 Enhanced 最终 Top-K 中优先覆盖不同来源文件
    rag_diversify_candidate_multiplier: int = 3  # 多样性选择前保留的精排候选倍数

    # ------------------------------------------------------------------
    # 评估 Judge 配置（独立于线上 RAG 模型，确保评估可复现）
    # ------------------------------------------------------------------
    eval_judge_model: str = "qwen3.5-plus"
    eval_judge_temperature: float = 0.0
    eval_judge_api_base: str = ""   # 空则复用 DASHSCOPE_API_BASE
    eval_judge_api_key: str = ""    # 空则复用 DASHSCOPE_API_KEY

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # Redis 配置（可选，不配置则使用 MemorySaver）
    redis_url: str = ""  # 如 "redis://localhost:6379"

    # 上下文裁剪配置
    context_max_tokens: int = 8000   # 上下文窗口 token 上限
    context_trimming_strategy: Literal["token_count", "none"] = "token_count"

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
