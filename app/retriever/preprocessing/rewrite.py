"""QueryRewritePreprocessor — rewrite 模式，使用 ChatQwen 改写查询

目标：将口语化、模糊的用户问题改写成更利于向量检索的规范表述。
使用 temperature=0 以保证改写结果稳定。
"""

from loguru import logger

from app.retriever.preprocessing.base import BaseQueryPreprocessor

_REWRITE_PROMPT_TEMPLATE = """\
你是一名专业的信息检索优化专家。请将下面的用户问题改写为更适合向量数据库语义检索的表述。

改写要求：
1. 保留原始问题的核心意图，不要改变问题的含义
2. 补充可能缺失的关键词和专业术语
3. 将口语化表述转化为书面化、规范化的表述
4. 输出只包含改写后的问题，不需要解释或多余内容

用户原始问题：{query}

改写后的问题："""


class QueryRewritePreprocessor(BaseQueryPreprocessor):
    """查询改写预处理器：使用 ChatQwen LLM 对查询进行语义增强改写

    懒初始化 LLM，仅在首次调用 process() 时创建，避免导入时副作用。
    """

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        """懒初始化 ChatQwen（temperature=0，快速稳定输出）"""
        if self._llm is None:
            from langchain_community.chat_models import ChatTongyi

            from app.config import config

            self._llm = ChatTongyi(
                model=config.rag_model,
                temperature=0,
                dashscope_api_key=config.dashscope_api_key,
            )
            logger.debug("[Preprocessor:rewrite] ChatQwen LLM 初始化完成")
        return self._llm

    def process(self, query: str) -> str:
        """使用 LLM 改写查询

        如果改写失败（网络超时、API 错误等），静默回退到原始查询，
        保证 RetrievalPipeline 不因预处理异常而中断。

        Args:
            query: 原始查询字符串

        Returns:
            str: 改写后的查询；失败时返回原始查询
        """
        try:
            prompt = _REWRITE_PROMPT_TEMPLATE.format(query=query)
            response = self._get_llm().invoke(prompt)
            rewritten = response.content.strip()

            if not rewritten:
                logger.warning("[Preprocessor:rewrite] LLM 返回空内容，回退到原始查询")
                return query

            logger.debug(
                f"[Preprocessor:rewrite] 改写完成\n"
                f"  原始: '{query[:80]}'\n"
                f"  改写: '{rewritten[:80]}'"
            )
            return rewritten

        except Exception as e:
            logger.warning(f"[Preprocessor:rewrite] 改写失败，回退到原始查询: {e}")
            return query
