"""LLM 辅助知识库文档生成脚本

基于现有 aiops-docs 作为 few-shot 示例，按场景 taxonomy 生成新运维 SOP 文档。

用法（在项目根目录执行）:

  python -m tests.evaluation.generate_docs                    # 生成全部场景
  python -m tests.evaluation.generate_docs --scene network    # 只生成指定场景
  python -m tests.evaluation.generate_docs --dry-run          # 预览 prompt 不调用 LLM

生成结果保存到 aiops-docs/ 目录，文件名带 _generated 后缀供人工审核。
审核通过后去掉 _generated 后缀即可纳入知识库。

场景 Taxonomy
=============
运维场景按以下 5 大类组织，确保知识库覆盖真实 AIOps 故障域：

  1. 资源告警 (resource)
     CPU、磁盘、内存 — 基础资源层面的异常检测与处理
     已覆盖: cpu_high_usage, disk_high_usage, memory_high_usage

  2. 服务可用性 (availability)
     服务实例健康状态、响应质量 — 面向业务可用性
     已覆盖: service_unavailable, slow_response

  3. 依赖故障 (dependency)
     数据库、消息队列、缓存等基础设施依赖的故障模式
     待覆盖: database_connection_pool, message_queue_backlog, cache_avalanche

  4. 链路异常 (connectivity)
     网络延迟、API 错误率 — 请求链路上的问题
     待覆盖: network_high_latency, api_error_rate_spike

  5. 容量/配置 (capacity_config)
     资源配额、证书生命周期 — 容量不足和配置漂移
     待覆盖: container_oom_killed, certificate_expiry

工具能力边界（生成文档时约束排查步骤）
======================================
生成文档中的排查步骤必须优先使用以下系统实际可用的工具：

  可用的 MCP 工具:
    - get_current_timestamp         获取当前毫秒时间戳
    - get_region_code_by_name       根据地区名查地区代码
    - get_topic_info_by_name        根据主题名查日志主题
    - search_topic_by_service_name  根据服务名查日志主题（支持模糊搜索）
    - search_log                    按时间范围和查询条件检索日志
    - query_cpu_metrics             查询 CPU 监控指标
    - query_memory_metrics          查询内存监控指标

  可用的内置工具:
    - get_current_time              获取当前日期时间字符串
    - retrieve_knowledge            从知识库检索相关文档

  暂不可用（标注为"未来能力"，不混入主排查步骤）:
    - kubectl / K8s API             容器编排操作
    - DB 直连 / SQL 执行            数据库直接操作
    - Redis CLI                     缓存直接操作
    - ping / traceroute / netstat   网络诊断命令
    - SSH 远程执行                  主机命令执行
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# 场景 Taxonomy 定义
# ---------------------------------------------------------------------------
SCENARIO_TAXONOMY = {
    "resource": {
        "label": "资源告警",
        "covered": ["cpu_high_usage", "disk_high_usage", "memory_high_usage"],
        "pending": [],
    },
    "availability": {
        "label": "服务可用性",
        "covered": ["service_unavailable", "slow_response"],
        "pending": [],
    },
    "dependency": {
        "label": "依赖故障",
        "covered": [],
        "pending": [
            {
                "scene_key": "database_connection_pool_exhaustion",
                "file_name": "database_connection_pool_exhaustion.md",
                "title": "数据库连接池耗尽告警处理方案",
                "alert_name": "DatabaseConnectionPoolExhaustion",
                "severity": "严重",
                "trigger": "数据库连接池活跃连接数持续5分钟超过90%",
                "description": "数据库连接池（如 HikariCP、Druid）连接耗尽导致新请求无法获取连接，表现为超时、请求堆积、服务雪崩。",
            },
            {
                "scene_key": "message_queue_backlog",
                "file_name": "message_queue_backlog.md",
                "title": "消息队列积压告警处理方案",
                "alert_name": "MessageQueueBacklog",
                "severity": "严重",
                "trigger": "消息队列（Kafka/RocketMQ）消费延迟持续10分钟超过10000条",
                "description": "消费者处理速度落后于生产速度，导致消息堆积、消费延迟、业务延迟。涉及 Kafka consumer lag 或 RocketMQ 消费位点。",
            },
            {
                "scene_key": "cache_avalanche",
                "file_name": "cache_avalanche.md",
                "title": "缓存雪崩/击穿告警处理方案",
                "alert_name": "CacheAvalanche",
                "severity": "紧急",
                "trigger": "缓存命中率骤降30%以上且数据库QPS飙升3倍",
                "description": "大量缓存 key 同时过期或热点 key 失效，请求直接穿透到数据库，导致 DB 压力剧增、服务响应变慢。",
            },
        ],
    },
    "connectivity": {
        "label": "链路异常",
        "covered": [],
        "pending": [
            {
                "scene_key": "network_high_latency",
                "file_name": "network_high_latency.md",
                "title": "网络延迟过高告警处理方案",
                "alert_name": "NetworkHighLatency",
                "severity": "警告",
                "trigger": "服务间网络延迟 P99 持续5分钟超过500ms",
                "description": "网络延迟导致服务间调用超时、请求堆积。可能由带宽饱和、丢包、路由异常引起。",
            },
            {
                "scene_key": "api_error_rate_spike",
                "file_name": "api_error_rate_spike.md",
                "title": "API 错误率飙升告警处理方案",
                "alert_name": "APIErrorRateSpike",
                "severity": "紧急",
                "trigger": "API 5xx 错误率持续3分钟超过5%",
                "description": "接口返回大量 5xx 错误，可能由上游依赖故障、代码缺陷、配置错误或流量峰值引起。",
            },
        ],
    },
    "capacity_config": {
        "label": "容量/配置",
        "covered": [],
        "pending": [
            {
                "scene_key": "container_oom_killed",
                "file_name": "container_oom_killed.md",
                "title": "容器 OOM 被杀告警处理方案",
                "alert_name": "ContainerOOMKilled",
                "severity": "紧急",
                "trigger": "容器被 OOM Killer 终止（exit code 137）",
                "description": "K8s Pod 因超出内存 limit 被 OOMKilled，容器重启、服务短暂不可用。排查内存配置、内存泄漏、流量突增。",
            },
            {
                "scene_key": "certificate_expiry",
                "file_name": "certificate_expiry.md",
                "title": "TLS 证书过期告警处理方案",
                "alert_name": "CertificateExpiry",
                "severity": "紧急",
                "trigger": "TLS 证书距离过期不足7天",
                "description": "TLS/SSL 证书即将过期或已过期，导致服务间 TLS 握手失败、用户浏览器安全警告。需紧急续期和轮换。",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# 工具能力清单（约束生成）
# ---------------------------------------------------------------------------
TOOL_CAPABILITIES = {
    "available": [
        {"name": "get_current_timestamp", "server": "CLS", "desc": "获取当前毫秒时间戳"},
        {"name": "get_region_code_by_name", "server": "CLS", "desc": "根据地区名查地区代码"},
        {"name": "get_topic_info_by_name", "server": "CLS", "desc": "根据主题名查日志主题"},
        {"name": "search_topic_by_service_name", "server": "CLS", "desc": "根据服务名模糊搜索日志主题"},
        {"name": "search_log", "server": "CLS", "desc": "按时间范围和查询条件检索日志"},
        {"name": "query_cpu_metrics", "server": "Monitor", "desc": "查询 CPU 监控指标"},
        {"name": "query_memory_metrics", "server": "Monitor", "desc": "查询内存监控指标"},
        {"name": "get_current_time", "server": "Built-in", "desc": "获取当前日期时间字符串"},
        {"name": "retrieve_knowledge", "server": "Built-in", "desc": "从知识库检索相关文档"},
    ],
    "unavailable": [
        "kubectl / K8s API（容器编排操作）",
        "DB 直连 / SQL 执行（数据库直接操作）",
        "Redis CLI（缓存直接操作）",
        "ping / traceroute / netstat（网络诊断命令）",
        "SSH 远程执行（主机命令执行）",
    ],
}

# ---------------------------------------------------------------------------
# Few-shot 加载
# ---------------------------------------------------------------------------
def _load_few_shot_docs(docs_dir: str = "aiops-docs") -> str:
    """加载现有文档作为 few-shot 示例"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return ""

    parts = []
    for md_file in sorted(docs_path.glob("*.md")):
        # 跳过已生成的待审核文件
        if "_generated" in md_file.name:
            continue
        content = md_file.read_text(encoding="utf-8")
        parts.append(f"### 参考文档: {md_file.name}\n\n{content}\n")

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个资深运维专家（SRE），擅长撰写运维排查 SOP 文档。

请参考以下示例文档的格式和风格，撰写一个新的运维排查 SOP 文档。

## 格式要求
严格按以下章节组织，使用 Markdown：

1. # {{场景名称}} — 文档标题
2. ## 告警名称 — 告警名、级别、触发条件
3. ## 问题描述 — 该故障会导致什么后果
4. ## 排查步骤 — 分步骤，每步标注使用的工具和参数
5. ## 常见原因分析 — 3-5 个常见原因，每个含特征 + 处理方案
6. ## 紧急处理措施 — 立即操作(5分钟) + 短期措施(30分钟) + 长期优化
7. ## 验证步骤 — 确认问题已解决的检查项
8. ## 相关告警 — 关联的其他告警
9. ## 联系方式 — 运维/开发团队联系方式

## 工具约束
排查步骤中的工具调用必须使用以下可用工具：
{available_tools}

以下工具不可用，排查步骤中不要依赖它们：
{unavailable_tools}

对于确实需要不可用工具的步骤（如 kubectl 操作），在文档末尾增加 "## 未来能力" 章节说明，
不要混入主排查步骤。

## 风格要求
- 使用中文
- 包含具体的工具名称和参数示例（如查询语句、地域名、日志主题名）
- 保持与示例文档一致的详细程度
- 每个常见原因要有"特征"和"处理方案"两部分
- 排查步骤要用 **工具**: `tool_name` 的格式标注"""


def _build_prompt(scene: dict, existing_docs: str) -> str:
    """构建文档生成 prompt"""
    available_str = "\n".join(
        f"  - `{t['name']}` ({t['server']}): {t['desc']}" for t in TOOL_CAPABILITIES["available"]
    )
    unavailable_str = "\n".join(f"  - {t}" for t in TOOL_CAPABILITIES["unavailable"])

    system = SYSTEM_PROMPT.format(
        available_tools=available_str,
        unavailable_tools=unavailable_str,
    )

    user = f"""## 新文档要求

场景: {scene['title']}
告警名: {scene['alert_name']}
告警级别: {scene['severity']}
触发条件: {scene['trigger']}
场景描述: {scene['description']}

## 已有的参考文档示例

{existing_docs}

请生成 {scene['file_name']} 的完整内容，直接输出 Markdown，不要加额外说明。"""

    return system, user


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------
def _get_llm():
    """延迟初始化 LLM"""
    from langchain_community.chat_models.tongyi import ChatTongyi

    # 延迟导入避免循环依赖
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import config

    kwargs = {
        "model": config.dashscope_model,
        "temperature": 0.3,
        "dashscope_api_key": config.dashscope_api_key,
    }
    api_base = os.environ.get("DASHSCOPE_API_BASE", "")
    if api_base:
        kwargs["dashscope_api_base"] = api_base
    return ChatTongyi(**kwargs)


def _generate_one(scene: dict, docs_dir: str, dry_run: bool = False) -> Optional[str]:
    """生成单个文档"""
    output_path = Path(docs_dir) / scene["file_name"].replace(".md", "_generated.md")

    existing_docs = _load_few_shot_docs(docs_dir)
    system_msg, user_msg = _build_prompt(scene, existing_docs)

    if dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN — 场景: {scene['title']}")
        print(f"输出文件: {output_path}")
        print(f"{'='*70}")
        print(f"\n[System Prompt 摘要] {len(system_msg)} chars")
        print(f"\n[User Prompt 前 500 chars]:\n{user_msg[:500]}...")
        return None

    llm = _get_llm()
    messages = [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"  已生成: {output_path} ({len(content)} chars)")
    return str(output_path)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_docs(
    docs_dir: str = "aiops-docs",
    scene_filter: Optional[str] = None,
    dry_run: bool = False,
):
    """生成所有待覆盖场景的文档

    Args:
        docs_dir: 文档输出目录
        scene_filter: 可选，只生成匹配分类的场景（如 "dependency"）
        dry_run: 仅预览 prompt，不调用 LLM
    """
    pending = []
    for cat_key, cat_info in SCENARIO_TAXONOMY.items():
        if scene_filter and scene_filter not in cat_key:
            continue
        for scene in cat_info["pending"]:
            scene["_category"] = cat_key
            pending.append(scene)

    if not pending:
        print("没有待生成的场景。")
        return

    print(f"场景 Taxonomy 覆盖: {len(pending)} 个待生成文档")
    print(f"分类分布: ", end="")
    cat_counts = {}
    for s in pending:
        cat_counts[s["_category"]] = cat_counts.get(s["_category"], 0) + 1
    print(", ".join(f"{SCENARIO_TAXONOMY[k]['label']}={v}" for k, v in cat_counts.items()))
    print(f"可用工具: {len(TOOL_CAPABILITIES['available'])} 个")
    print(f"不可用工具: {len(TOOL_CAPABILITIES['unavailable'])} 个")
    print()

    for i, scene in enumerate(pending, 1):
        cat_label = SCENARIO_TAXONOMY[scene["_category"]]["label"]
        print(f"[{i}/{len(pending)}] {cat_label} → {scene['title']}")
        _generate_one(scene, docs_dir, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 辅助知识库文档生成")
    parser.add_argument(
        "--scene", "-s", type=str, default=None,
        help="只生成指定分类: resource, availability, dependency, connectivity, capacity_config",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览 prompt，不调用 LLM",
    )
    parser.add_argument(
        "--docs-dir", type=str, default="aiops-docs",
        help="文档输出目录（默认: aiops-docs）",
    )
    args = parser.parse_args()

    generate_docs(docs_dir=args.docs_dir, scene_filter=args.scene, dry_run=args.dry_run)
