"""RAG 评估数据集

手工构建 + LLM 辅助生成的 Q&A 数据集，用于 RAGAs 评估。
覆盖多篇 aiops-docs 文档，按场景 taxonomy 组织。

数据契约（EvalSample）：
  必填: question, ground_truths, relevant_docs
  可选: category, reference_docs

问题分类（category）：
  - exact_keyword : 精确关键词/技术术语查询
  - colloquial    : 口语化查询
  - cross_doc     : 跨文档综合查询

relevant_docs 说明：
  存储与当前问题相关的源文档文件名列表（如 ["cpu_high_usage.md"]）。
  用于 Hit Rate@k 和 MRR 计算时，与检索结果的 doc.metadata["_file_name"] 做交集匹配。
  跨文档问题可包含多个文件名。

ground_truth 拼接规则：
  每个样本的 ground_truths 是 list[str]，在构建 Dataset 时通过 "\n" 连接为
  单个字符串。RAGAs 期望 ground_truth 为字符串形式。

-------------------------------------------------------------------------------
标注规范（Annotation Guidelines）
===============================================================================

1. relevant_docs 判定标准
...............................................................................

  直接相关（必填）:
    问题的答案直接来源于该文档。该文档包含了回答问题所需的核心信息。
    示例: Q="CPU飙高怎么排查" → relevant_docs=["cpu_high_usage.md"]

  部分相关（选填，建议标注）:
    问题涉及该文档的部分内容，但主要答案在其他文档中。
    示例: Q="哪些告警会导致服务不可用" → 可能涉及 cpu/disk/memory/service_unavailable
          此时标注所有涉及的文档

  背景相关（不标注）:
    文档提到了该概念，但不是答案的直接来源。避免过度标注导致检索
    评估失真。

  判定原则:
    - 优先标注答案的"直接来源"文档
    - 跨文档问题必须标注所有直接相关的文档（2-3 个）
    - 如有歧义，在 question 上方加注释说明判定理由

2. ground_truths 粒度标准
...............................................................................

  数量: 每个问题 3-5 个要点
  长度: 每个要点 1-2 句话（20-60 字）
  内容: 覆盖"是什么 → 为什么 → 怎么办"
    - 是什么: 问题的现象或原因
    - 为什么: 产生该现象的原因
    - 怎么办: 具体的处理步骤或命令

  示例（好的 ground_truth）:
    Q: "CPU飙高可能是代码里写了死循环吗？"
    ground_truths: [
      "可能是死循环或无限递归导致。这是常见原因之一。",
      "表现为某个线程CPU使用率持续在100%左右。",
      "应该使用 jstack 或 gdb 等工具抓取线程堆栈，分析处于 RUNNABLE 状态的线程",
    ]

  反例（太简单/太冗长）:
    BAD: ["会。"]                                          # 太短，无实质内容
    BAD: ["首先登录服务器，然后执行 top 命令..." (200字)]    # 太长，应拆成多个要点

3. 标注审核流程
...............................................................................

  生成阶段 (Generation):
    LLM 按文档和问题类型模板生成候选问题 + 初步 ground_truths + relevant_docs

  初审 (Initial Review):
    人工检查: 问题是否可答、relevant_docs 是否准确、ground_truths 是否达标
    标记: 通过 / 需修改 / 淘汰

  复审 (Re-review):
    抽查初审通过的问题，确认标注一致性（不同人标注同一问题的结果应接近）

  冻结 (Freeze):
    确认后加入 EVALUATION_DATASET，递增 DATASET_VERSION
    冻结后修改问题需更新版本号并在提交信息中说明变更原因

4. 质量控制规则
...............................................................................

  去重:
    - 相似问题（意思相同仅措辞不同）只保留一个
    - 使用 embedding 余弦相似度检测，阈值 > 0.85 的视为重复

  覆盖率:
    - 每个文档至少 5 个问题
    - 每个问题类型（exact_keyword/colloquial/cross_doc）至少占总数的 10%
    - 新增数据前先运行 validate_testset() 确认增量覆盖了什么能力缺口

  回归基线:
    - 每次扩充数据集后，运行完整评估并记录基线分数
    - 如果原有问题的 Hit Rate 下降 > 5%，检查是否新文档干扰了检索排序
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# 数据集版本号 — 修改测试集内容后递增
# ---------------------------------------------------------------------------
DATASET_VERSION = "1.1.1"


@dataclass
class EvalSample:
    """单条评估样本的数据契约

    Attributes:
        question:       用户查询文本（必填）
        ground_truths:  期望参考答案要点列表（必填）
        relevant_docs:  相关源文档文件名列表（必填），如 ["cpu_high_usage.md"]
        category:       问题分类标签，用于分组统计
        reference_docs: 参考文档来源（可选）
    """

    question: str
    ground_truths: List[str]
    relevant_docs: List[str] = field(default_factory=list)
    category: str = "exact_keyword"
    reference_docs: List[str] = field(default_factory=list)


def validate_testset(samples: List[EvalSample]) -> List[str]:
    """评估前校验数据集完整性和一致性

    Returns:
        校验错误信息列表，空列表表示通过
    """
    errors: List[str] = []

    if not samples:
        errors.append("数据集为空，至少需要一条样本")
        return errors

    valid_categories = {"exact_keyword", "colloquial", "cross_doc"}

    for i, s in enumerate(samples):
        prefix = f"[样本 {i}]"

        if not s.question or not s.question.strip():
            errors.append(f"{prefix} question 为空或仅含空白字符")

        if not s.ground_truths:
            errors.append(f"{prefix} ground_truths 为空列表")
        else:
            for j, gt in enumerate(s.ground_truths):
                if not gt or not gt.strip():
                    errors.append(f"{prefix} ground_truths[{j}] 为空或仅含空白字符")

        if not s.relevant_docs:
            errors.append(f"{prefix} relevant_docs 为空列表，需标注至少一个相关文档")

        if s.category not in valid_categories:
            errors.append(
                f"{prefix} category='{s.category}' 不在有效值 {valid_categories} 中"
            )

    return errors


# ---------------------------------------------------------------------------
# 评估数据集（25 题，覆盖 5 篇文档）
# ---------------------------------------------------------------------------

EVALUATION_DATASET: List[EvalSample] = [
    # ---------------------------------------------
    # CPU 使用率过高告警处理方案 (cpu_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="CPU 告警后，我要怎么查是哪个进程在吃 CPU？",
        ground_truths=[
            "使用 top -c 命令按 CPU 使用率排序",
            "使用 ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -10 获取 Top 10 CPU 进程",
            "使用 pidstat 1 5 获取进程的实时 CPU 统计信息",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="CPU飙高可能是代码里写了死循环吗？",
        ground_truths=[
            "可能是死循环或无限递归导致。这是常见原因之一。",
            "表现为某个线程CPU使用率持续在100%左右。",
            "应该使用 jstack 或 gdb 等工具抓取线程堆栈，分析处于 RUNNABLE 状态的线程",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="遇到CPU100%怎么紧急处理？限流还是扩容？",
        ground_truths=[
            "如果是流量突增导致，如果影响核心链路且无法自动扩容，应立即开启限流降级",
            "同时申请紧急扩容增加实例数",
            "如果是死循环导致的，且影响核心链路，应当立刻重启相关实例",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="数据库查询慢会拖慢应用服务器的CPU吗？",
        ground_truths=[
            "会。数据库查询慢会导致应用层大量线程阻塞，上下文切换频繁，从而导致 CPU 使用率升高。",
            "处理方法是通知 DBA 排查慢查询，或在应用侧紧急降级非核心查询",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="排查 CPU 问题时怎么看应用日志有没有报错？",
        ground_truths=[
            "在应用日志中搜索 ERROR 或 Exception 关键字",
            "特别关注 OutOfMemoryError、TimeoutException 等异常",
            "还要关注是否存在大量重复的错误日志",
        ],
        relevant_docs=["cpu_high_usage.md"],
        category="colloquial",
    ),

    # ---------------------------------------------
    # 磁盘使用率过高告警处理方案 (disk_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="怎么看哪个文件夹把磁盘占满了？",
        ground_truths=[
            "执行 df -h 查看各分区使用情况，找出使用率超过告警阈值的磁盘分区",
            "进入目标分区，使用 du -sh * | sort -hr 找出占用空间最大的目录",
            "可以使用 find /path -type f -size +500M 查找大于500MB的大文件",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="日志把磁盘写满了，可以直接清空吗？怎么操作？",
        ground_truths=[
            "不要直接使用 rm 删除正在被程序写入的日志文件",
            "应使用 echo '' > application.log 或 > application.log 清空文件内容，以保留文件句柄",
            "可以直接删除 N 天前的旧日志文件：find /var/log -type f -mtime +7 -name '*.log' -delete",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="磁盘高是不是因为 Docker 镜像太多？怎么清理？",
        ground_truths=[
            "是的，Docker镜像和无用容器积累会占用大量磁盘空间。",
            "如果是 Docker 占用，可以执行 docker system prune -a --volumes 清理无用数据",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="磁盘告警紧急处理的30分钟内措施是什么？",
        ground_truths=[
            "压缩旧的大文件：gzip old_file.log",
            "排查并清理临时文件目录 /tmp 或 /var/tmp",
            "如果系统存在大文件传输或导入，暂停这些非紧急批处理任务",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="如果磁盘 inode 被占满了怎么排查？",
        ground_truths=[
            "使用 df -i 命令查看 inode 的使用情况",
            "如果 inode 满了（即使磁盘空间还有剩余），通常是因为有大量小文件",
            "需要查找包含大量小文件的目录并清理",
        ],
        relevant_docs=["disk_high_usage.md"],
        category="exact_keyword",
    ),

    # ---------------------------------------------
    # 内存使用率过高告警处理方案 (memory_high_usage.md)
    # ---------------------------------------------
    EvalSample(
        question="内存满了怎么抓堆栈分析？是打 dump 吗？",
        ground_truths=[
            "是的，使用 jmap -dump:live,format=b,file=heap.bin <PID> 生成堆转储文件",
            "然后使用 MAT (Memory Analyzer Tool) 或 JProfiler 进行离线分析",
            "通过分析大对象或实例数最多的类来定位内存泄漏",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="OOM是不是因为缓存配置不对？",
        ground_truths=[
            "可能是缓存配置不当导致的。",
            "如果本地缓存（如 Guava Cache、Caffeine）未设置合理的过期时间或最大容量，会导致缓存无限增长",
            "或者一次性从数据库加载了过大的缓存预热数据",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="大文件处理会导致内存高吗？",
        ground_truths=[
            "是的，一次性读取过大的文件（如 CSV、Excel）到内存中会导致对象激增",
            "或者在内存中进行了大批量的集合操作",
            "建议优化为流式处理或分页处理",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="内存告警 5 分钟内我该做啥操作？",
        ground_truths=[
            "如果影响核心链路，应当立刻隔离异常节点（摘除流量），避免雪崩",
            "在节点挂掉或重启前，务必抓取现场：执行 jstat 或保留 OOM 时自动生成的 heap dump",
            "如果是严重内存泄漏且无法快速修复，执行应用重启",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="colloquial",
    ),
    EvalSample(
        question="怎么看 JVM 的内存使用详情？",
        ground_truths=[
            "使用 jstat -gcutil <PID> 1000 查看实时 GC 和堆内存使用比例",
            "使用 jmap -heap <PID> 查看堆内存的配置和使用详情",
            "如果频繁出现 Full GC，说明老年代内存不足或存在内存泄漏",
        ],
        relevant_docs=["memory_high_usage.md"],
        category="exact_keyword",
    ),

    # ---------------------------------------------
    # 服务不可用告警处理方案 (service_unavailable.md)
    # ---------------------------------------------
    EvalSample(
        question="接口不通了，服务挂了怎么看进程还在不在？",
        ground_truths=[
            "使用 ps -ef | grep <应用名> 检查服务进程是否存在",
            "使用 netstat -tlnp | grep <端口号> 检查监听端口是否正常",
            "如果进程存在但端口不通，可能是线程池耗尽或死锁",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
    ),
    EvalSample(
        question="服务不可用可能是因为数据库连不上吗？",
        ground_truths=[
            "是的。数据库连接失败、连接池满、数据库本身宕机等都会导致服务不可用。",
            "表现为大量请求阻塞，应用日志中出现 SQLTimeoutException 或 Connection refused 错误",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
    ),
    EvalSample(
        question="出现服务不可用告警，1 分钟内必须要干嘛？",
        ground_truths=[
            "确认告警真实性，访问服务的健康检查接口 (/health)",
            "如果有备用集群或跨机房容灾，立即通知运维进行流量切换",
            "如果是新版本发布导致的，立即执行版本回滚",
        ],
        relevant_docs=["service_unavailable.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="怎么确认是不是外部依赖服务挂了导致的不可用？",
        ground_truths=[
            "检查调用链监控或日志，看是否有大量调用外部依赖超时的报错",
            "如果有熔断器（如 Sentinel, Hystrix），检查是否已经触发熔断",
            "手动 curl 或 ping 外部依赖服务的端点进行连通性测试",
        ],
        relevant_docs=["service_unavailable.md"],
        category="colloquial",
    ),
    EvalSample(
        question="服务不可用事件结束后需要复盘吗？",
        ground_truths=[
            "必须复盘。在故障恢复后 24 小时内组织相关人员复盘",
            "分析根本原因，输出故障报告",
            "制定改进项（如加强监控、优化重试策略等）并录入系统跟踪解决",
        ],
        relevant_docs=["service_unavailable.md"],
        category="cross_doc",
    ),

    # ---------------------------------------------
    # 服务响应时间过长告警处理方案 (slow_response.md)
    # ---------------------------------------------
    EvalSample(
        question="RT 升高、接口慢，怎么查是不是慢SQL导致的？",
        ground_truths=[
            "登录数据库管理平台或监控大盘，查看慢 SQL 统计",
            "或者在应用日志中搜索 Slow query 或耗时较长的 SQL 记录",
            "对疑似慢 SQL 执行 EXPLAIN 查看执行计划，确认是否命中索引或发生了全表扫描",
        ],
        relevant_docs=["slow_response.md"],
        category="colloquial",
    ),
    EvalSample(
        question="缓存击穿会导致接口变慢吗？",
        ground_truths=[
            "会。缓存穿透、击穿或雪崩会导致大量请求直接打到数据库",
            "数据库负载急剧上升，从而导致整体响应变慢",
            "监控表现为缓存命中率断崖式下跌，数据库 QPS 异常突增",
        ],
        relevant_docs=["slow_response.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="接口RT高，代码可能有啥问题？",
        ground_truths=[
            "代码中可能存在复杂的循环计算",
            "频繁的同步 IO 操作（如文件读写）",
            "或者在循环中调用外部 RPC 或查询数据库（N+1 查询问题）",
        ],
        relevant_docs=["slow_response.md"],
        category="colloquial",
    ),
    EvalSample(
        question="响应慢的问题，30 分钟内怎么处理？",
        ground_truths=[
            "如果是慢 SQL，紧急给涉及的表添加缺失的索引",
            "如果是下游依赖慢，且非核心链路，紧急开启或调低降级开关，熔断弱依赖",
            "如果是缓存失效，修复缓存逻辑并进行缓存预热",
        ],
        relevant_docs=["slow_response.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="怎样预防接口变慢？",
        ground_truths=[
            "梳理所有外部依赖，配置合理的超时时间和重试机制",
            "对核心接口实施严格的限流和降级策略",
            "所有上线的新 SQL 必须经过 DBA 审核，确保执行计划正确",
        ],
        relevant_docs=["slow_response.md"],
        category="cross_doc",
    ),

    # -------------------------------------------------------
    # 批次: candidate_questions_20260522_205717.json（导入于 2026-05-22 21:17）
    # -------------------------------------------------------
    EvalSample(
        question="APIErrorRateSpike 告警的触发条件是什么？",
        ground_truths=[
            "API 5xx 错误率持续3分钟超过5%",
            "告警级别为紧急",
            "触发后可能导致用户请求失败和业务中断",
            "应通过 search_log 查询日志确认是上游依赖、代码缺陷、配置错误还是流量峰值导致",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="接口返回大量 5xx 错误怎么办？",
        ground_truths=[
            "确定告警发生的时间范围，用于后续日志查询",
            "搜索服务对应的日志主题并检查日志主题信息",
            "按时间范围和查询条件检索日志，分析错误日志",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
    ),
    EvalSample(
        question="怎么处理 API 错误率突然升高的问题？",
        ground_truths=[
            "获取当前时间以确定告警发生的时间范围",
            "使用 search_topic_by_service_name 工具搜索服务对应的日志主题",
            "按时间范围和查询条件检索日志，分析错误日志",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
    ),
    EvalSample(
        question="API 错误率飙升可能是哪些原因导致的？",
        ground_truths=[
            "上游依赖故障：日志中有上游服务调用失败记录",
            "代码缺陷：特定代码路径频繁抛出异常",
            "配置错误：最近有配置变更，日志中有配置加载错误",
            "流量峰值：请求量突然激增，响应时间变长但无明显错误",
            "网络问题：无法访问服务，网络连接超时",
        ],
        relevant_docs=["api_error_rate_spike.md"],
        category="colloquial",
    ),
    EvalSample(
        question="CacheAvalanche 告警的触发条件是什么？",
        ground_truths=[
            "缓存命中率骤降30%以上且数据库QPS飙升3倍",
            "大量缓存 key 同时过期或热点 key 失效，请求穿透到数据库",
            "触发后数据库压力剧增，可能引发服务雪崩",
            "需通过 search_log 查询缓存命中率日志和数据库 QPS 确认根因",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="缓存突然失效了怎么办？",
        ground_truths=[
            "立即预热缓存，重启后立即预热热点数据",
            "优化缓存策略，设置合理的过期时间",
            "增加缓存容量，考虑扩容缓存",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
    ),
    EvalSample(
        question="怎么排查缓存雪崩问题？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询缓存服务日志主题，使用 search_topic_by_service_name 工具",
            "按时间检索缓存命中率日志，使用 search_log 工具",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
    ),
    EvalSample(
        question="缓存命中率低会影响什么？",
        ground_truths=[
            "数据库压力剧增",
            "服务响应变慢",
            "可能引发系统雪崩效应",
            "用户体验下降",
        ],
        relevant_docs=["cache_avalanche.md"],
        category="colloquial",
    ),
    EvalSample(
        question="网络延迟高会影响缓存吗？",
        ground_truths=[
            "网络故障可能导致缓存服务器不可达",
            "网络延迟高会导致缓存命中率骤降",
            "日志中会有网络超时错误",
        ],
        relevant_docs=["cache_avalanche.md", "network_high_latency.md"],
        category="edge_case",
    ),
    EvalSample(
        question="CertificateExpiry 告警的触发条件是什么？",
        ground_truths=[
            "TLS 证书距离过期不足7天",
            "告警级别为紧急",
            "触发后服务间 TLS 握手将失败，用户浏览器显示安全警告",
            "需立即续签证书、更新服务器配置并重启相关服务",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="search_log 工具在查询系统日志时需要哪些参数？",
        ground_truths=[
            "地域: ap-guangzhou",
            "日志主题: system-metrics",
            "时间范围: 最近30分钟",
            "查询条件: event:certificate_expiry OR level:ERROR",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="SSL 证书快到期了怎么办？",
        ground_truths=[
            "立即使用证书管理工具（如Let's Encrypt）续签证书。",
            "更新证书文件到服务器并重启相关服务。",
            "设置证书到期前30天的提醒，定期检查证书有效期。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
    ),
    EvalSample(
        question="网站访问时浏览器显示安全警告怎么处理？",
        ground_truths=[
            "检查TLS/SSL证书是否即将过期或已过期。",
            "如果证书过期，立即续签证书并更新到服务器。",
            "验证新证书是否正确安装，并测试SSL握手是否成功。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
    ),
    EvalSample(
        question="应用出现 SSL 握手失败怎么排查？",
        ground_truths=[
            "查询系统日志和应用日志，确认证书的有效期和状态。",
            "检查证书路径和配置文件，确保证书链完整。",
            "如果证书过期，立即续签并更新到服务器。",
        ],
        relevant_docs=["certificate_expiry.md"],
        category="colloquial",
    ),
    EvalSample(
        question="网络延迟高会影响 SSL 证书吗？",
        ground_truths=[
            "网络延迟高不会直接影响SSL证书的有效性。",
            "但高延迟可能会导致SSL握手过程变慢，影响用户体验。",
            "建议检查网络状况，优化网络配置。",
        ],
        relevant_docs=["certificate_expiry.md", "network_high_latency.md"],
        category="edge_case",
    ),
    EvalSample(
        question="ContainerOOMKilled 告警的触发条件是什么？",
        ground_truths=[
            "容器被 OOM Killer 终止（exit code 137）",
            "K8s Pod 因超出内存 limit 被 OOMKilled",
            "触发后容器重启，服务短暂不可用，可能导致数据丢失",
            "需通过 query_memory_metrics 和容器日志确认内存使用情况和 OOM 根因",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="query_memory_metrics 工具需要传哪些参数？",
        ground_truths=[
            "地域: ap-guangzhou",
            "时间范围: 最近30分钟",
            "查询条件: 内存使用率超过90%",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="如果容器突然挂了，怎么排查是不是因为内存问题？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询系统监控日志，检查内存使用率是否超过90%",
            "查询容器日志，查找 level:ERROR 或 level:WARN 或 event:OOM 的记录",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="colloquial",
    ),
    EvalSample(
        question="应用突然崩溃了，怎么知道是不是内存不够用了？",
        ground_truths=[
            "查看应用日志中是否有 OutOfMemoryError 或 OOM 相关错误记录",
            "使用 query_memory_metrics 查看内存使用趋势，确认是否存在持续上升或突增",
            "检查应用进程的退出码，exit code 137 表示被 OOM Killer 终止",
        ],
        relevant_docs=["container_oom_killed.md"],
        category="colloquial",
    ),
    EvalSample(
        question="数据库连接池耗尽会影响容器内存吗？",
        ground_truths=[
            "数据库连接池耗尽通常不会直接导致容器内存问题",
            "但连接等待队列堆积会间接增加应用线程数和内存占用",
            "需同时检查数据库连接池状态和容器内存使用情况，交叉确认因果关系",
        ],
        relevant_docs=["container_oom_killed.md", "database_connection_pool_exhaustion.md"],
        category="edge_case",
    ),
    EvalSample(
        question="DatabaseConnectionPoolExhaustion 告警的触发条件是什么？",
        ground_truths=[
            "数据库连接池活跃连接数持续5分钟超过90%",
            "告警级别为严重",
            "触发后新请求无法获取数据库连接，导致请求超时增加",
            "需通过 search_log 查询连接池日志，分析活跃连接数和等待队列长度",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="遇到数据库连接池耗尽怎么办？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "查询系统日志和数据库连接池日志，分析连接池状态",
            "查询应用日志，分析请求日志和数据库性能指标",
            "根据常见原因分析并采取相应处理方案",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
    ),
    EvalSample(
        question="数据库连接池满了怎么排查？",
        ground_truths=[
            "获取当前时间，确定告警发生的时间范围",
            "使用 search_topic_by_service_name 查询系统日志",
            "检查数据库连接池日志，提取连接池状态信息",
            "查询应用日志，分析请求量变化和错误堆栈信息",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
    ),
    EvalSample(
        question="连接池满了会导致什么问题？",
        ground_truths=[
            "新请求无法获取数据库连接",
            "请求超时增加",
            "服务响应变慢",
            "可能触发雪崩效应",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md"],
        category="colloquial",
    ),
    EvalSample(
        question="网络延迟高会影响数据库连接池吗？",
        ground_truths=[
            "网络延迟高可能导致数据库连接不稳定",
            "日志中可能会有网络超时错误",
            "建议将数据库与应用部署在同一可用区减少网络跳数",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "network_high_latency.md"],
        category="edge_case",
    ),
    EvalSample(
        question="消息积压了怎么办？",
        ground_truths=[
            "增加消费者实例数量",
            "启用生产者限流机制",
            "优化消费者处理逻辑",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
    ),
    EvalSample(
        question="怎么检查消息队列的日志？",
        ground_truths=[
            "使用 search_topic_by_service_name 工具查询日志主题",
            "使用 get_topic_info_by_name 工具获取具体的日志主题信息",
            "使用 search_log 工具检索相关日志",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
    ),
    EvalSample(
        question="消息队列积压后，如何分析消费者的CPU和内存使用情况？",
        ground_truths=[
            "使用 query_cpu_metrics 和 query_memory_metrics 工具查询消费者服务的资源使用情况",
            "参数包括服务名（消费者所在的服务名）和时间范围（最近1小时）",
            "如果 CPU 或内存使用率超过 80%，说明消费者可能存在资源瓶颈，需扩容或优化代码",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="colloquial",
    ),
    EvalSample(
        question="磁盘满了会影响消息队列吗？",
        ground_truths=[
            "消息队列 Broker 需要磁盘存储消息，磁盘满会直接导致消息写入失败",
            "磁盘 IO 繁忙也会拖慢消息写入和消费速度，加剧积压",
            "应使用 search_log 查询消息队列日志中是否有磁盘空间不足或 IO 超时的错误",
        ],
        relevant_docs=["message_queue_backlog.md"],
        category="edge_case",
    ),
    EvalSample(
        question="NetworkHighLatency 告警的触发条件是什么？",
        ground_truths=[
            "服务间网络延迟 P99 持续5分钟超过500ms",
            "告警级别为警告",
            "触发后服务间调用可能超时，请求堆积，用户体验下降",
            "应通过 search_log 查询延迟日志，确认受影响的服务对，通知网络团队排查链路",
        ],
        relevant_docs=["network_high_latency.md"],
        category="exact_keyword",
    ),
    EvalSample(
        question="网络延迟过高怎么排查？",
        ground_truths=[
            "使用 search_log 查询 network-metrics 日志主题，确认延迟最高的服务对",
            "通过应用日志中的 RPC 调用耗时记录确认受影响的服务间调用",
            "使用 query_cpu_metrics 和 query_memory_metrics 排除服务器资源瓶颈",
            "联系网络团队检查跨地域链路的带宽和延迟",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
    ),
    EvalSample(
        question="服务调用变慢了，怎么定位问题？",
        ground_truths=[
            "获取当前时间并确定告警发生的时间范围",
            "使用 search_log 查询系统日志和应用日志，分析网络延迟和错误记录",
            "使用 query_cpu_metrics 和 query_memory_metrics 确认是否因服务器资源瓶颈导致",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
    ),
    EvalSample(
        question="网络延迟高会导致哪些问题？",
        ground_truths=[
            "可能导致服务间调用超时。",
            "请求堆积，用户体验下降。",
            "可能触发雪崩效应。",
        ],
        relevant_docs=["network_high_latency.md"],
        category="colloquial",
    ),
    EvalSample(
        question="数据库连接池耗尽会影响网络延迟吗？",
        ground_truths=[
            "数据库连接池耗尽可能导致应用线程阻塞等待连接，间接增加请求处理延迟",
            "大量阻塞线程可能占满应用线程池，导致新的网络请求无法被处理",
            "需通过应用日志确认连接等待时长，同时检查网络延迟是否由其他因素引起",
        ],
        relevant_docs=["database_connection_pool_exhaustion.md", "network_high_latency.md"],
        category="edge_case",
    ),
]


def get_eval_dataset():
    """将 EvalSample 列表转换为 RAGAs 评估所需的 Dataset 格式

    RAGAs 期望字段：
      - question      (str)
      - ground_truth  (str) — 多个 ground_truths 通过 "\n" 拼接为单个字符串

    Returns:
        datasets.Dataset: 包含 question, ground_truth, category, relevant_docs 的 Dataset
    """
    from datasets import Dataset

    questions = []
    ground_truths = []
    categories = []
    relevant_docs_list = []

    for s in EVALUATION_DATASET:
        questions.append(s.question)
        ground_truths.append("\n".join(s.ground_truths))
        categories.append(s.category)
        relevant_docs_list.append(s.relevant_docs)

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "category": categories,
        "relevant_docs": relevant_docs_list,
    })
