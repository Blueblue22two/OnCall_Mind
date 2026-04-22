"""PassthroughPreprocessor — none 模式，原样透传查询"""

from loguru import logger

from app.retriever.preprocessing.base import BaseQueryPreprocessor


class PassthroughPreprocessor(BaseQueryPreprocessor):
    """透传预处理器：不做任何改写，直接返回原始查询"""

    def process(self, query: str) -> str:
        logger.debug(f"[Preprocessor:none] 透传查询: '{query[:60]}'")
        return query
