"""Milvus 客户端工厂模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    connections,
    utility,
    MilvusException,
)

from app.config import config


def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """
    langchain_milvus 内部创建的 MilvusClient 会将 _using 设为 ``cm-{id}``，
    该别名未在 pymilvus.orm.connections 中注册；随后 ORM ``Collection(..., using=...)``
    会抛出 ConnectionNotExistException: should create connection first.

    在已通过 ``connections.connect(alias="default", ...)`` 建立连接后，
    强制让 MilvusClient 使用 ``default`` 别名，与 ORM 一致。
    """
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        return

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    MilvusClient.__init__ = _wrapped_init  # type: ignore[method-assign]
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    ENHANCED_COLLECTION_NAME: str = "biz_enhanced"  # Phase 2 增强集合
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None
        self._enhanced_collection: Collection | None = None  # Phase 2 集合句柄

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        # 幂等：导入阶段可能已由 VectorStoreManager 等提前连接，避免重复初始化
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立连接
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建客户端
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)
                
                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break
                
                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.COLLECTION_NAME}'...")
                        _ = utility.drop_collection(self.COLLECTION_NAME)
                        logger.info(f"正在重新创建 collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection
            self._load_collection()

            # 初始化 biz_enhanced collection（enhanced 模式所需）
            self._init_enhanced_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 biz collection 是否存在"""
        # pymilvus 的类型标注可能不准确，实际返回 bool
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)  # type: ignore[arg-type]

    def _enhanced_collection_exists(self) -> bool:
        """检查 biz_enhanced collection 是否存在"""
        result = utility.has_collection(self.ENHANCED_COLLECTION_NAME)
        return bool(result)  # type: ignore[arg-type]

    def _init_enhanced_collection(self) -> None:
        """初始化 biz_enhanced collection（幂等）"""
        try:
            if not self._enhanced_collection_exists():
                logger.info(f"collection '{self.ENHANCED_COLLECTION_NAME}' 不存在，正在创建...")
                self._create_enhanced_collection()
                logger.info(f"成功创建 collection '{self.ENHANCED_COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.ENHANCED_COLLECTION_NAME}' 已存在")
                self._enhanced_collection = Collection(self.ENHANCED_COLLECTION_NAME)

            self._load_enhanced_collection()
        except Exception as e:
            # enhanced collection 初始化失败不应阻断基础功能
            logger.warning(f"biz_enhanced collection 初始化失败（enhanced 模式不可用）: {e}")

    def _create_enhanced_collection(self) -> None:
        """创建 biz_enhanced collection（双向量 schema + Milvus 内置 BM25）

        Schema 设计：
          - id            VARCHAR PK
          - dense_vector  FLOAT_VECTOR(1024, COSINE)  — DashScope 嵌入
          - content_text  VARCHAR(8000, analyzer=jieba) — BM25 输入字段
          - sparse_vector SPARSE_FLOAT_VECTOR         — 由内置 BM25 Function 自动生成
          - metadata      JSON                         — 文档元数据
        """
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            # BM25 输入字段，需要启用分析器（Jieba 中文分词）
            FieldSchema(
                name="content_text",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},  # Milvus 内置 Jieba 中文分析器
            ),
            # 稀疏向量由 BM25 Function 自动填充，无需客户端写入
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # Milvus 内置 BM25：从 content_text 自动生成 sparse_vector
        bm25_function = Function(
            name="biz_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content_text"],
            output_field_names=["sparse_vector"],
        )

        schema = CollectionSchema(
            fields=fields,
            functions=[bm25_function],
            description="Business knowledge collection with hybrid dense+sparse vectors",
            enable_dynamic_field=False,
        )

        self._enhanced_collection = Collection(
            name=self.ENHANCED_COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        self._create_enhanced_index()

    def _create_enhanced_index(self) -> None:
        """为 biz_enhanced 的 dense_vector 和 sparse_vector 创建索引"""
        if self._enhanced_collection is None:
            raise RuntimeError("Enhanced Collection 未初始化")

        # Dense 向量索引（COSINE + IVF_FLAT）
        dense_index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        _ = self._enhanced_collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params,
        )
        logger.info("成功为 dense_vector 创建索引（COSINE）")

        # 稀疏向量索引（BM25 必须用 SPARSE_INVERTED_INDEX）
        sparse_index_params = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
        }
        _ = self._enhanced_collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params,
        )
        logger.info("成功为 sparse_vector 创建索引（BM25 SPARSE_INVERTED_INDEX）")

    def _load_enhanced_collection(self) -> None:
        """加载 biz_enhanced collection 到内存"""
        if self._enhanced_collection is None:
            self._enhanced_collection = Collection(self.ENHANCED_COLLECTION_NAME)

        try:
            load_state = utility.load_state(self.ENHANCED_COLLECTION_NAME)
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._enhanced_collection.load()
                logger.info(f"成功加载 collection '{self.ENHANCED_COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.ENHANCED_COLLECTION_NAME}' 已加载")
        except AttributeError:
            try:
                self._enhanced_collection.load()
                logger.info(f"成功加载 collection '{self.ENHANCED_COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.ENHANCED_COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 enhanced collection 失败: {e}")
            raise

    def get_enhanced_collection(self) -> Collection:
        """获取 biz_enhanced collection 实例

        Returns:
            Collection: enhanced collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._enhanced_collection is None:
            raise RuntimeError("Enhanced Collection 未初始化，请先调用 connect()")
        return self._enhanced_collection

    def _create_collection(self) -> None:
        """创建 biz collection"""
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        # 创建 collection
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为 vector 字段创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            "metric_type": "L2",  # 欧氏距离
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        _ = self._collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        # 检查 collection 是否已加载（兼容多版本）
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            # 尝试列出 connections
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if self._enhanced_collection is not None:
                self._enhanced_collection.release()
                self._enhanced_collection = None
        except Exception as e:
            errors.append(f"释放 enhanced collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None
        
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
milvus_manager = MilvusClientManager()
