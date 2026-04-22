"""RAG 评估数据集

手工构建的 25 条 Q&A 数据集，用于 RAGAs 评估。
覆盖 5 篇 aiops-docs 文档，每篇 5 题。

结构：
  - question: 用户查询（包含一些口语化、口径不一的测试点以验证改写效果）
  - ground_truths: 期望的标准答案关键信息列表（供 RAGAs 计算指标用）
"""

EVALUATION_DATASET = [
    # ---------------------------------------------
    # CPU 使用率过高告警处理方案 (cpu_high_usage.md)
    # ---------------------------------------------
    {
        "question": "CPU 告警后，我要怎么查是哪个进程在吃 CPU？",
        "ground_truths": [
            "使用 top -c 命令按 CPU 使用率排序",
            "使用 ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -10 获取 Top 10 CPU 进程",
            "使用 pidstat 1 5 获取进程的实时 CPU 统计信息"
        ],
    },
    {
        "question": "CPU飙高可能是代码里写了死循环吗？",
        "ground_truths": [
            "可能是死循环或无限递归导致。这是常见原因之一。",
            "表现为某个线程CPU使用率持续在100%左右。",
            "应该使用 jstack 或 gdb 等工具抓取线程堆栈，分析处于 RUNNABLE 状态的线程"
        ],
    },
    {
        "question": "遇到CPU100%怎么紧急处理？限流还是扩容？",
        "ground_truths": [
            "如果是流量突增导致，如果影响核心链路且无法自动扩容，应立即开启限流降级",
            "同时申请紧急扩容增加实例数",
            "如果是死循环导致的，且影响核心链路，应当立刻重启相关实例"
        ],
    },
    {
        "question": "数据库查询慢会拖慢应用服务器的CPU吗？",
        "ground_truths": [
            "会。数据库查询慢会导致应用层大量线程阻塞，上下文切换频繁，从而导致 CPU 使用率升高。",
            "处理方法是通知 DBA 排查慢查询，或在应用侧紧急降级非核心查询"
        ],
    },
    {
        "question": "排查 CPU 问题时怎么看应用日志有没有报错？",
        "ground_truths": [
            "在应用日志中搜索 ERROR 或 Exception 关键字",
            "特别关注 OutOfMemoryError、TimeoutException 等异常",
            "还要关注是否存在大量重复的错误日志"
        ],
    },

    # ---------------------------------------------
    # 磁盘使用率过高告警处理方案 (disk_high_usage.md)
    # ---------------------------------------------
    {
        "question": "怎么看哪个文件夹把磁盘占满了？",
        "ground_truths": [
            "执行 df -h 查看各分区使用情况，找出使用率超过告警阈值的磁盘分区",
            "进入目标分区，使用 du -sh * | sort -hr 找出占用空间最大的目录",
            "可以使用 find /path -type f -size +500M 查找大于500MB的大文件"
        ],
    },
    {
        "question": "日志把磁盘写满了，可以直接清空吗？怎么操作？",
        "ground_truths": [
            "不要直接使用 rm 删除正在被程序写入的日志文件",
            "应使用 echo '' > application.log 或 > application.log 清空文件内容，以保留文件句柄",
            "可以直接删除 N 天前的旧日志文件：find /var/log -type f -mtime +7 -name '*.log' -delete"
        ],
    },
    {
        "question": "磁盘高是不是因为 Docker 镜像太多？怎么清理？",
        "ground_truths": [
            "是的，Docker镜像和无用容器积累会占用大量磁盘空间。",
            "如果是 Docker 占用，可以执行 docker system prune -a --volumes 清理无用数据"
        ],
    },
    {
        "question": "磁盘告警紧急处理的30分钟内措施是什么？",
        "ground_truths": [
            "压缩旧的大文件：gzip old_file.log",
            "排查并清理临时文件目录 /tmp 或 /var/tmp",
            "如果系统存在大文件传输或导入，暂停这些非紧急批处理任务"
        ],
    },
    {
        "question": "如果磁盘 inode 被占满了怎么排查？",
        "ground_truths": [
            "使用 df -i 命令查看 inode 的使用情况",
            "如果 inode 满了（即使磁盘空间还有剩余），通常是因为有大量小文件",
            "需要查找包含大量小文件的目录并清理"
        ],
    },

    # ---------------------------------------------
    # 内存使用率过高告警处理方案 (memory_high_usage.md)
    # ---------------------------------------------
    {
        "question": "内存满了怎么抓堆栈分析？是打 dump 吗？",
        "ground_truths": [
            "是的，使用 jmap -dump:live,format=b,file=heap.bin <PID> 生成堆转储文件",
            "然后使用 MAT (Memory Analyzer Tool) 或 JProfiler 进行离线分析",
            "通过分析大对象或实例数最多的类来定位内存泄漏"
        ],
    },
    {
        "question": "OOM是不是因为缓存配置不对？",
        "ground_truths": [
            "可能是缓存配置不当导致的。",
            "如果本地缓存（如 Guava Cache、Caffeine）未设置合理的过期时间或最大容量，会导致缓存无限增长",
            "或者一次性从数据库加载了过大的缓存预热数据"
        ],
    },
    {
        "question": "大文件处理会导致内存高吗？",
        "ground_truths": [
            "是的，一次性读取过大的文件（如 CSV、Excel）到内存中会导致对象激增",
            "或者在内存中进行了大批量的集合操作",
            "建议优化为流式处理或分页处理"
        ],
    },
    {
        "question": "内存告警 5 分钟内我该做啥操作？",
        "ground_truths": [
            "如果影响核心链路，应当立刻隔离异常节点（摘除流量），避免雪崩",
            "在节点挂掉或重启前，务必抓取现场：执行 jstat 或保留 OOM 时自动生成的 heap dump",
            "如果是严重内存泄漏且无法快速修复，执行应用重启"
        ],
    },
    {
        "question": "怎么看 JVM 的内存使用详情？",
        "ground_truths": [
            "使用 jstat -gcutil <PID> 1000 查看实时 GC 和堆内存使用比例",
            "使用 jmap -heap <PID> 查看堆内存的配置和使用详情",
            "如果频繁出现 Full GC，说明老年代内存不足或存在内存泄漏"
        ],
    },

    # ---------------------------------------------
    # 服务不可用告警处理方案 (service_unavailable.md)
    # ---------------------------------------------
    {
        "question": "接口不通了，服务挂了怎么看进程还在不在？",
        "ground_truths": [
            "使用 ps -ef | grep <应用名> 检查服务进程是否存在",
            "使用 netstat -tlnp | grep <端口号> 检查监听端口是否正常",
            "如果进程存在但端口不通，可能是线程池耗尽或死锁"
        ],
    },
    {
        "question": "服务不可用可能是因为数据库连不上吗？",
        "ground_truths": [
            "是的。数据库连接失败、连接池满、数据库本身宕机等都会导致服务不可用。",
            "表现为大量请求阻塞，应用日志中出现 SQLTimeoutException 或 Connection refused 错误"
        ],
    },
    {
        "question": "出现服务不可用告警，1 分钟内必须要干嘛？",
        "ground_truths": [
            "确认告警真实性，访问服务的健康检查接口 (/health)",
            "如果有备用集群或跨机房容灾，立即通知运维进行流量切换",
            "如果是新版本发布导致的，立即执行版本回滚"
        ],
    },
    {
        "question": "怎么确认是不是外部依赖服务挂了导致的不可用？",
        "ground_truths": [
            "检查调用链监控或日志，看是否有大量调用外部依赖超时的报错",
            "如果有熔断器（如 Sentinel, Hystrix），检查是否已经触发熔断",
            "手动 curl 或 ping 外部依赖服务的端点进行连通性测试"
        ],
    },
    {
        "question": "服务不可用事件结束后需要复盘吗？",
        "ground_truths": [
            "必须复盘。在故障恢复后 24 小时内组织相关人员复盘",
            "分析根本原因，输出故障报告",
            "制定改进项（如加强监控、优化重试策略等）并录入系统跟踪解决"
        ],
    },

    # ---------------------------------------------
    # 服务响应时间过长告警处理方案 (slow_response.md)
    # ---------------------------------------------
    {
        "question": "RT 升高、接口慢，怎么查是不是慢SQL导致的？",
        "ground_truths": [
            "登录数据库管理平台或监控大盘，查看慢 SQL 统计",
            "或者在应用日志中搜索 Slow query 或耗时较长的 SQL 记录",
            "对疑似慢 SQL 执行 EXPLAIN 查看执行计划，确认是否命中索引或发生了全表扫描"
        ],
    },
    {
        "question": "缓存击穿会导致接口变慢吗？",
        "ground_truths": [
            "会。缓存穿透、击穿或雪崩会导致大量请求直接打到数据库",
            "数据库负载急剧上升，从而导致整体响应变慢",
            "监控表现为缓存命中率断崖式下跌，数据库 QPS 异常突增"
        ],
    },
    {
        "question": "接口RT高，代码可能有啥问题？",
        "ground_truths": [
            "代码中可能存在复杂的循环计算",
            "频繁的同步 IO 操作（如文件读写）",
            "或者在循环中调用外部 RPC 或查询数据库（N+1 查询问题）"
        ],
    },
    {
        "question": "响应慢的问题，30 分钟内怎么处理？",
        "ground_truths": [
            "如果是慢 SQL，紧急给涉及的表添加缺失的索引",
            "如果是下游依赖慢，且非核心链路，紧急开启或调低降级开关，熔断弱依赖",
            "如果是缓存失效，修复缓存逻辑并进行缓存预热"
        ],
    },
    {
        "question": "怎样预防接口变慢？",
        "ground_truths": [
            "梳理所有外部依赖，配置合理的超时时间和重试机制",
            "对核心接口实施严格的限流和降级策略",
            "所有上线的新 SQL 必须经过 DBA 审核，确保执行计划正确"
        ],
    }
]

def get_eval_dataset():
    """将上述字典列表转换为 RAGAs 需要的数据格式

    RAGAs 所需的基础字段：
      - question (list[str])
      - ground_truth (list[str]): 注意新版 ragas 中通常用 `ground_truth`（字符串形式，而不是列表的列表），
        为兼容不同版本，这里把 ground_truths 列表合并为一个段落。
    """
    from datasets import Dataset

    questions = []
    ground_truths = []

    for item in EVALUATION_DATASET:
        questions.append(item["question"])
        # 将多个真理点合并成一段文本
        truth_text = "\n".join(item["ground_truths"])
        ground_truths.append(truth_text)

    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths
    })
