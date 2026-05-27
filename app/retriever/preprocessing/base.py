"""查询预处理器抽象基类"""

from abc import ABC, abstractmethod


class BaseQueryPreprocessor(ABC):
    """查询预处理器统一接口

    所有预处理器必须实现 process() 方法。
    输入原始查询字符串，返回处理后的查询字符串。
    """

    @abstractmethod
    def process(self, query: str) -> str:
        """对查询文本进行预处理

        Args:
            query: 用户的原始查询字符串

        Returns:
            str: 处理后的查询字符串（可能与原始查询相同）
        """
        ...
