# OnCall Mind Python 版本 Makefile
# 用于自动化项目初始化、Docker 管理和文档向量化

# ============================================================
# 配置变量
# ============================================================
SERVER_URL = http://localhost:9900
UPLOAD_API = $(SERVER_URL)/api/upload
HEALTH_CHECK_API = $(SERVER_URL)/health
DOCS_DIR = aiops-docs
MILVUS_CONTAINER = milvus-standalone

# 颜色输出
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
CYAN = \033[0;36m
NC = \033[0m

.PHONY: help init start stop restart check upload clean up down status wait \
        install install-dev dev run test test-quick format lint fix type-check \
        security pre-commit-install pre-commit check-all coverage docs shell \
        ipython watch add add-dev remove list-docs test-upload sync logs \
        start-cls stop-cls start-monitor stop-monitor start-api stop-api status-mcp \
        docker-build docker-up docker-down docker-logs docker-status docker-push \
        docker-push-versioned docker-shell index-rag-v4-basic index-rag-v4-raw index-rag-v4-prefix \
        index-rag-v4-child eval-rag-v4-clean-cache eval-rag-v4-raw \
        eval-rag-v4-prefix eval-rag-v4-child eval-rag-v4-parent \
        eval-rag-v4-adaptive eval-rag-v4-priors eval-rag-v4-ablation \
        eval-rag-v4-ablation-resume eval-rag-v4-basic eval-rag-v4-gate \
        eval-rag-v4-test

# ============================================================
# 默认目标：显示帮助信息
# ============================================================
help:
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  OnCall Mind Python 版本 - Makefile 命令$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(CYAN)【一键操作】$(NC)"
	@echo "  $(YELLOW)make init$(NC)         - 🚀 一键初始化（Docker → 服务 → 上传文档）"
	@echo ""
	@echo "$(CYAN)【Docker 管理】$(NC)"
	@echo "  $(YELLOW)make up$(NC)           - 🐳 启动 Milvus 容器"
	@echo "  $(YELLOW)make down$(NC)         - 🛑 停止 Milvus 容器"
	@echo "  $(YELLOW)make status$(NC)       - 📊 查看容器状态"
	@echo ""
	@echo "$(CYAN)【服务管理】$(NC)"
	@echo "  $(YELLOW)make start$(NC)        - 🚀 启动所有服务（MCP + FastAPI）"
	@echo "  $(YELLOW)make stop$(NC)         - 🛑 停止所有服务（MCP + FastAPI）"
	@echo "  $(YELLOW)make restart$(NC)      - 🔄 重启所有服务"
	@echo "  $(YELLOW)make check$(NC)        - 🔍 检查 FastAPI 服务状态"
	@echo "  $(YELLOW)make status-mcp$(NC)   - 📊 查看 MCP 服务状态"
	@echo ""
	@echo "$(CYAN)【MCP 服务管理】$(NC)"
	@echo "  $(YELLOW)make start-cls$(NC)     - 📋 启动 CLS MCP 服务"
	@echo "  $(YELLOW)make stop-cls$(NC)      - 🛑 停止 CLS MCP 服务"
	@echo "  $(YELLOW)make start-monitor$(NC) - 📊 启动 Monitor MCP 服务"
	@echo "  $(YELLOW)make stop-monitor$(NC)  - 🛑 停止 Monitor MCP 服务"
	@echo "  $(YELLOW)make start-api$(NC)     - 🚀 启动 FastAPI 服务"
	@echo "  $(YELLOW)make stop-api$(NC)      - 🛑 停止 FastAPI 服务"
	@echo ""
	@echo "$(CYAN)【开发模式】$(NC)"
	@echo "  $(YELLOW)make dev$(NC)          - 🔧 开发模式运行（前台，热重载）"
	@echo "  $(YELLOW)make run$(NC)          - 🏭 生产模式运行（前台）"
	@echo ""
	@echo "$(CYAN)【文档管理】$(NC)"
	@echo "  $(YELLOW)make upload$(NC)       - 📤 上传 docs 目录下的文档"
	@echo "  $(YELLOW)make list-docs$(NC)    - 📚 列出可上传的文档"
	@echo "  $(YELLOW)make test-upload$(NC)  - 🧪 测试上传单个文件"
	@echo ""
	@echo "$(CYAN)【依赖管理】$(NC)"
	@echo "  $(YELLOW)make install$(NC)      - 📦 安装生产依赖"
	@echo "  $(YELLOW)make install-dev$(NC)  - 📦 安装开发依赖"
	@echo "  $(YELLOW)make sync$(NC)         - 🔄 同步依赖"
	@echo "  $(YELLOW)make add PKG=xxx$(NC)  - ➕ 添加依赖包"
	@echo ""
	@echo "$(CYAN)【代码质量】$(NC)"
	@echo "  $(YELLOW)make format$(NC)       - 🎨 格式化代码"
	@echo "  $(YELLOW)make lint$(NC)         - 🔍 代码检查"
	@echo "  $(YELLOW)make fix$(NC)          - 🔧 自动修复问题"
	@echo "  $(YELLOW)make test$(NC)         - 🧪 运行测试"
	@echo "  $(YELLOW)make check-all$(NC)    - ✅ 运行所有检查"
	@echo ""
	@echo "$(CYAN)【RAG 评估】$(NC)"
	@echo "  $(YELLOW)make eval-rag$(NC)            - 🧪 运行 RAG 检索评估"
	@echo "  $(YELLOW)make eval-rag-full$(NC)       - 🧪 完整 RAG 评估（检索+生成）"
	@echo "  $(YELLOW)make eval-generation$(NC)     - 🧪 生成质量独立评估"
	@echo "  $(YELLOW)make eval-compare$(NC)        - 📊 对比 basic vs enhanced"
	@echo "  $(YELLOW)make eval-ablation$(NC)       - 🔬 消融实验"
	@echo "  $(YELLOW)make eval-validate-dataset$(NC) - 🔍 验证数据集质量"
	@echo "  $(YELLOW)make eval-baseline-save$(NC)  - 📌 保存当前结果为基线"
	@echo "  $(YELLOW)make eval-baseline-check$(NC) - 📊 与基线对比检查退化"
	@echo "  $(YELLOW)make eval-rag-v4-ablation$(NC) - 🔬 构建隔离索引并运行 v4 分块消融"
	@echo ""
	@echo "$(CYAN)【其他】$(NC)"
	@echo "  $(YELLOW)make clean$(NC)        - 🧹 清理临时文件"
	@echo "  $(YELLOW)make shell$(NC)        - 🐍 启动 Python Shell"
	@echo "  $(YELLOW)make coverage$(NC)     - 📊 查看测试覆盖率"
	@echo "  $(YELLOW)make logs$(NC)         - 📜 查看服务日志"
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)使用示例:$(NC)"
	@echo "  1. 一键初始化: $(YELLOW)make init$(NC)"
	@echo "  2. 启动服务:   $(YELLOW)make start$(NC) (自动启动 CLS + Monitor MCP + FastAPI)"
	@echo "  3. 检查状态:   $(YELLOW)make status-mcp$(NC)"
	@echo "  4. 停止服务:   $(YELLOW)make stop$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"

# ============================================================
# 一键初始化
# ============================================================
init:
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)🚀 开始一键初始化 OnCall Mind...$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)步骤 1/4: 启动 Docker 容器（Milvus 向量数据库）$(NC)"
	@$(MAKE) up
	@echo ""
	@echo "$(YELLOW)步骤 2/4: 启动 FastAPI 服务$(NC)"
	@$(MAKE) start
	@echo ""
	@echo "$(YELLOW)步骤 3/4: 等待服务就绪$(NC)"
	@$(MAKE) wait
	@echo ""
	@echo "$(YELLOW)步骤 4/4: 上传文档到向量数据库$(NC)"
	@$(MAKE) upload
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✅ 初始化完成！所有文档已成功向量化存储到数据库$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)🌐 服务访问地址:$(NC)"
	@echo "   API 服务: $(SERVER_URL)"
	@echo "   API 文档: $(SERVER_URL)/docs"
	@echo "   Attu (Milvus Web UI): http://localhost:8000"
	@echo "   MinIO: http://localhost:9001 (admin/minioadmin)"
	@echo ""
	@echo "$(YELLOW)💡 提示: 服务正在后台运行$(NC)"
	@echo "   查看日志: $(YELLOW)tail -f server.log$(NC)"
	@echo "   停止服务: $(YELLOW)make stop$(NC)"

# ============================================================
# Docker 管理
# ============================================================

# 启动 Docker 容器（使用 docker compose）
up:
	@echo "$(YELLOW)🐳 检查 Docker 容器状态...$(NC)"
	@if ! docker info > /dev/null 2>&1; then \
		echo "$(YELLOW)⚠️  Docker 未运行，尝试启动 Colima...$(NC)"; \
		colima start 2>/dev/null || (echo "$(RED)❌ 无法启动 Docker，请手动启动$(NC)" && exit 1); \
		sleep 3; \
	fi
	@if docker ps --format '{{.Names}}' | grep -q "^$(MILVUS_CONTAINER)$$"; then \
		echo "$(GREEN)✅ Milvus 容器已经在运行中$(NC)"; \
		docker ps --filter "name=milvus" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -10; \
	else \
		echo "$(YELLOW)🚀 启动 Milvus 相关容器...$(NC)"; \
		docker compose -f vector-database.yml up -d; \
		echo "$(YELLOW)⏳ 等待容器启动...$(NC)"; \
		sleep 5; \
		if docker ps --format '{{.Names}}' | grep -q "^$(MILVUS_CONTAINER)$$"; then \
			echo "$(GREEN)✅ Docker 容器启动成功！$(NC)"; \
			echo ""; \
			echo "$(GREEN)📋 运行中的容器:$(NC)"; \
			docker ps --filter "name=milvus" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -10; \
			echo ""; \
			echo "$(GREEN)🌐 服务访问地址:$(NC)"; \
			echo "   Milvus: localhost:19530"; \
			echo "   Attu (Web UI): http://localhost:8000"; \
			echo "   MinIO: http://localhost:9001 (admin/minioadmin)"; \
		else \
			echo "$(RED)❌ 容器启动失败$(NC)"; \
			exit 1; \
		fi; \
	fi

# 停止 Docker 容器
down:
	@echo "$(YELLOW)🛑 停止 Docker 容器...$(NC)"
	@if docker ps --format '{{.Names}}' | grep -q "milvus"; then \
		docker compose -f vector-database.yml down; \
		echo "$(GREEN)✅ Docker 容器已停止$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  没有运行中的 Milvus 容器$(NC)"; \
	fi

# 查看容器状态
status:
	@echo "$(YELLOW)📊 Docker 容器状态:$(NC)"
	@echo ""
	@if docker ps -a --format '{{.Names}}' | grep -q "milvus"; then \
		docker ps -a --filter "name=milvus" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo ""; \
		running=$$(docker ps --filter "name=milvus" --format '{{.Names}}' | wc -l | tr -d ' '); \
		total=$$(docker ps -a --filter "name=milvus" --format '{{.Names}}' | wc -l | tr -d ' '); \
		echo "$(GREEN)运行中: $$running / $$total$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  没有找到 Milvus 相关容器$(NC)"; \
		echo "$(YELLOW)提示: 请先创建 Milvus 容器$(NC)"; \
	fi

# ============================================================
# MCP 服务管理
# ============================================================

# 启动 CLS MCP 服务
start-cls:
	@echo "$(YELLOW)📋 启动 CLS MCP 服务...$(NC)"
	@if pgrep -f "mcp_servers/cls_server.py" > /dev/null 2>&1; then \
		echo "$(GREEN)✅ CLS MCP 服务已经在运行中$(NC)"; \
	else \
		echo "$(YELLOW)📦 正在启动 CLS MCP 服务（后台运行）...$(NC)"; \
		nohup .venv/bin/python mcp_servers/cls_server.py > mcp_cls.log 2>&1 & \
		echo $$! > mcp_cls.pid; \
		sleep 2; \
		if pgrep -f "mcp_servers/cls_server.py" > /dev/null 2>&1; then \
			echo "$(GREEN)✅ CLS MCP 服务启动成功$(NC)"; \
			echo "$(YELLOW)   PID: $$(cat mcp_cls.pid)$(NC)"; \
			echo "$(YELLOW)   URL: http://127.0.0.1:8003/mcp$(NC)"; \
			echo "$(YELLOW)   日志: mcp_cls.log$(NC)"; \
		else \
			echo "$(RED)❌ CLS MCP 服务启动失败$(NC)"; \
			echo "$(YELLOW)请检查日志: tail -f mcp_cls.log$(NC)"; \
		fi; \
	fi

# 启动 Monitor MCP 服务
start-monitor:
	@echo "$(YELLOW)📊 启动 Monitor MCP 服务...$(NC)"
	@if pgrep -f "mcp_servers/monitor_server.py" > /dev/null 2>&1; then \
		echo "$(GREEN)✅ Monitor MCP 服务已经在运行中$(NC)"; \
	else \
		echo "$(YELLOW)📦 正在启动 Monitor MCP 服务（后台运行）...$(NC)"; \
		nohup .venv/bin/python mcp_servers/monitor_server.py > mcp_monitor.log 2>&1 & \
		echo $$! > mcp_monitor.pid; \
		sleep 2; \
		if pgrep -f "mcp_servers/monitor_server.py" > /dev/null 2>&1; then \
			echo "$(GREEN)✅ Monitor MCP 服务启动成功$(NC)"; \
			echo "$(YELLOW)   PID: $$(cat mcp_monitor.pid)$(NC)"; \
			echo "$(YELLOW)   URL: http://127.0.0.1:8004/mcp$(NC)"; \
			echo "$(YELLOW)   日志: mcp_monitor.log$(NC)"; \
		else \
			echo "$(RED)❌ Monitor MCP 服务启动失败$(NC)"; \
			echo "$(YELLOW)请检查日志: tail -f mcp_monitor.log$(NC)"; \
		fi; \
	fi

# 停止 Monitor MCP 服务
stop-monitor:
	@echo "$(YELLOW)🛑 停止 Monitor MCP 服务...$(NC)"
	@if [ -f mcp_monitor.pid ]; then \
		pid=$$(cat mcp_monitor.pid); \
		if ps -p $$pid > /dev/null 2>&1; then \
			kill $$pid; \
			echo "$(GREEN)✅ Monitor MCP 服务已停止 (PID: $$pid)$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  进程不存在 (PID: $$pid)$(NC)"; \
		fi; \
		rm -f mcp_monitor.pid; \
	else \
		echo "$(YELLOW)⚠️  未找到 mcp_monitor.pid 文件$(NC)"; \
		pkill -f "mcp_servers/monitor_server.py" 2>/dev/null && \
			echo "$(GREEN)✅ 已停止所有 Monitor MCP 进程$(NC)" || \
			echo "$(YELLOW)⚠️  没有运行中的 Monitor MCP 进程$(NC)"; \
	fi

# 检查 MCP 服务状态
status-mcp:
	@echo "$(YELLOW)📊 MCP 服务状态:$(NC)"
	@echo ""
	@echo "$(CYAN)CLS MCP 服务:$(NC)"
	@if pgrep -f "mcp_servers/cls_server.py" > /dev/null 2>&1; then \
		pid=$$(pgrep -f "mcp_servers/cls_server.py"); \
		echo "  状态: $(GREEN)运行中$(NC)"; \
		echo "  PID: $$pid"; \
		echo "  URL: http://127.0.0.1:8003/mcp"; \
		curl -s http://127.0.0.1:8003/mcp > /dev/null 2>&1 && \
			echo "  连接: $(GREEN)✅ 正常$(NC)" || \
			echo "  连接: $(RED)❌ 无法连接$(NC)"; \
	else \
		echo "  状态: $(RED)未运行$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN)Monitor MCP 服务:$(NC)"
	@if pgrep -f "mcp_servers/monitor_server.py" > /dev/null 2>&1; then \
		pid=$$(pgrep -f "mcp_servers/monitor_server.py"); \
		echo "  状态: $(GREEN)运行中$(NC)"; \
		echo "  PID: $$pid"; \
		echo "  URL: http://127.0.0.1:8004/mcp"; \
		curl -s http://127.0.0.1:8004/mcp > /dev/null 2>&1 && \
			echo "  连接: $(GREEN)✅ 正常$(NC)" || \
			echo "  连接: $(RED)❌ 无法连接$(NC)"; \
	else \
		echo "  状态: $(RED)未运行$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN)Math MCP 服务:$(NC)"
	@echo "  状态: $(YELLOW)已移除（示例服务）$(NC)"

# ============================================================
# FastAPI 服务管理
# ============================================================

# 启动所有服务（MCP + FastAPI）
start:
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)🚀 启动所有服务$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@$(MAKE) start-cls
	@sleep 1
	@echo ""
	@$(MAKE) start-monitor
	@sleep 1
	@echo ""
	@$(MAKE) start-api
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✅ 所有服务启动完成！$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"

# 启动 FastAPI 服务
start-api:
	@echo "$(YELLOW)🚀 启动 FastAPI 服务...$(NC)"
	@if curl -s -f $(HEALTH_CHECK_API) > /dev/null 2>&1; then \
		echo "$(GREEN)✅ FastAPI 服务已经在运行中 ($(SERVER_URL))$(NC)"; \
	else \
		echo "$(YELLOW)📦 正在启动 FastAPI 服务（后台运行）...$(NC)"; \
		nohup .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 9900 > server.log 2>&1 & \
		echo $$! > server.pid; \
		echo "$(GREEN)✅ FastAPI 服务启动命令已执行$(NC)"; \
		echo "$(YELLOW)   PID: $$(cat server.pid)$(NC)"; \
		echo "$(YELLOW)   URL: $(SERVER_URL)$(NC)"; \
		echo "$(YELLOW)   日志: server.log$(NC)"; \
	fi

# 停止所有服务（FastAPI + MCP）
stop:
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)🛑 停止所有服务$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@$(MAKE) stop-api
	@echo ""
	@$(MAKE) stop-cls
	@echo ""
	@$(MAKE) stop-monitor
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✅ 所有服务已停止！$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"

# 停止 CLS MCP 服务
stop-cls:
	@echo "$(YELLOW)🛑 停止 CLS MCP 服务...$(NC)"
	@if [ -f mcp_cls.pid ]; then \
		pid=$$(cat mcp_cls.pid); \
		if ps -p $$pid > /dev/null 2>&1; then \
			kill $$pid; \
			echo "$(GREEN)✅ CLS MCP 服务已停止 (PID: $$pid)$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  进程不存在 (PID: $$pid)$(NC)"; \
		fi; \
		rm -f mcp_cls.pid; \
	else \
		echo "$(YELLOW)⚠️  未找到 mcp_cls.pid 文件$(NC)"; \
		pkill -f "mcp_servers/cls_server.py" 2>/dev/null && \
			echo "$(GREEN)✅ 已停止所有 CLS MCP 进程$(NC)" || \
			echo "$(YELLOW)⚠️  没有运行中的 CLS MCP 进程$(NC)"; \
	fi

# 停止 FastAPI 服务
stop-api:
	@echo "$(YELLOW)🛑 停止 FastAPI 服务...$(NC)"
	@if [ -f server.pid ]; then \
		pid=$$(cat server.pid); \
		if ps -p $$pid > /dev/null 2>&1; then \
			kill $$pid; \
			echo "$(GREEN)✅ FastAPI 服务已停止 (PID: $$pid)$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  进程不存在 (PID: $$pid)$(NC)"; \
		fi; \
		rm -f server.pid; \
	else \
		echo "$(YELLOW)⚠️  未找到 server.pid 文件$(NC)"; \
		pkill -f "uvicorn app.main:app" 2>/dev/null && \
			echo "$(GREEN)✅ 已停止所有 uvicorn 进程$(NC)" || \
			echo "$(YELLOW)⚠️  没有运行中的 uvicorn 进程$(NC)"; \
	fi

# 重启所有服务
restart:
	@echo "$(YELLOW)🔄 重启所有服务...$(NC)"
	@echo ""
	@$(MAKE) stop
	@sleep 2
	@$(MAKE) start
	@$(MAKE) wait
	@echo ""
	@echo "$(GREEN)✅ 所有服务重启完成！$(NC)"

# 等待服务就绪（最多 60 秒）
wait:
	@echo "$(YELLOW)⏳ 等待服务器就绪...$(NC)"
	@max_attempts=60; \
	attempt=0; \
	while [ $$attempt -lt $$max_attempts ]; do \
		if curl -s -f $(HEALTH_CHECK_API) > /dev/null 2>&1; then \
			echo ""; \
			echo "$(GREEN)✅ 服务器已就绪！($(SERVER_URL))$(NC)"; \
			exit 0; \
		fi; \
		attempt=$$((attempt + 1)); \
		printf "\r$(YELLOW)   等待中... [$$attempt/$$max_attempts]$(NC)"; \
		sleep 1; \
	done; \
	echo ""; \
	echo "$(RED)❌ 服务器启动超时！$(NC)"; \
	echo "$(YELLOW)请检查日志: tail -f server.log$(NC)"; \
	exit 1

# 检查服务状态
check:
	@echo "$(YELLOW)🔍 检查服务器状态...$(NC)"
	@if curl -s -f $(HEALTH_CHECK_API) > /dev/null 2>&1; then \
		echo "$(GREEN)✅ 服务器运行正常 ($(SERVER_URL))$(NC)"; \
		echo ""; \
		echo "$(CYAN)健康检查响应:$(NC)"; \
		curl -s $(HEALTH_CHECK_API) | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))" 2>/dev/null || curl -s $(HEALTH_CHECK_API); \
	else \
		echo "$(RED)❌ 服务器未运行或无法连接！$(NC)"; \
		echo "$(YELLOW)请先启动服务: make start$(NC)"; \
		exit 1; \
	fi

# 开发模式运行（前台，热重载）
dev:
	@echo "$(YELLOW)🔧 启动开发服务器（热重载）...$(NC)"
	.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 9900

# 生产模式运行（前台）
run:
	@echo "$(YELLOW)🏭 启动生产服务器...$(NC)"
	.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# ============================================================
# 文档管理
# ============================================================

# 上传所有文档
upload:
	@echo "$(YELLOW)📤 开始上传 $(DOCS_DIR) 目录下的文档...$(NC)"
	@if [ ! -d "$(DOCS_DIR)" ]; then \
		echo "$(RED)❌ 目录 $(DOCS_DIR) 不存在！$(NC)"; \
		exit 1; \
	fi
	@count=0; \
	success=0; \
	failed=0; \
	for file in $(DOCS_DIR)/*.md; do \
		if [ -f "$$file" ]; then \
			count=$$((count + 1)); \
			filename=$$(basename "$$file"); \
			echo "$(YELLOW)  [$$count] 上传文件: $$filename$(NC)"; \
			response=$$(curl -s -w "\n%{http_code}" -X POST $(UPLOAD_API) \
				-F "file=@$$file" \
				-H "Accept: application/json"); \
			http_code=$$(echo "$$response" | tail -n1); \
			body=$$(echo "$$response" | sed '$$d'); \
			if [ "$$http_code" = "200" ]; then \
				echo "$(GREEN)      ✅ 成功: $$filename$(NC)"; \
				success=$$((success + 1)); \
			else \
				echo "$(RED)      ❌ 失败: $$filename (HTTP $$http_code)$(NC)"; \
				echo "$$body" | head -n 3; \
				failed=$$((failed + 1)); \
			fi; \
			sleep 1; \
		fi; \
	done; \
	echo ""; \
	echo "$(GREEN)📊 上传统计:$(NC)"; \
	echo "   总计: $$count 个文件"; \
	echo "   $(GREEN)成功: $$success$(NC)"; \
	if [ $$failed -gt 0 ]; then \
		echo "   $(RED)失败: $$failed$(NC)"; \
	fi

# 列出文档
list-docs:
	@echo "$(YELLOW)📚 $(DOCS_DIR) 目录下的文档:$(NC)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		ls -lh $(DOCS_DIR)/*.md 2>/dev/null || echo "$(RED)没有找到 .md 文件$(NC)"; \
	else \
		echo "$(RED)目录 $(DOCS_DIR) 不存在$(NC)"; \
	fi

# 测试上传单个文件
test-upload:
	@echo "$(YELLOW)🧪 测试上传单个文件...$(NC)"
	@first_file=$$(ls $(DOCS_DIR)/*.md 2>/dev/null | head -n1); \
	if [ -n "$$first_file" ]; then \
		echo "$(YELLOW)上传文件: $$first_file$(NC)"; \
		curl -X POST $(UPLOAD_API) \
			-F "file=@$$first_file" \
			-H "Accept: application/json" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))" 2>/dev/null || \
			curl -X POST $(UPLOAD_API) -F "file=@$$first_file"; \
	else \
		echo "$(RED)测试文件不存在$(NC)"; \
	fi

# ============================================================
# 依赖管理
# ============================================================

install:  ## 安装依赖（生产环境）
	@echo "$(YELLOW)📦 安装依赖...$(NC)"
	pip install -r requirements.txt 2>/dev/null || pip install -e .
	@echo "$(GREEN)✅ 依赖安装完成$(NC)"

install-dev:  ## 安装开发依赖
	@echo "$(YELLOW)📦 安装开发依赖...$(NC)"
	pip install -e ".[dev]" 2>/dev/null || pip install -e .
	@echo "$(GREEN)✅ 开发依赖安装完成$(NC)"

sync:  ## 同步依赖
	@echo "$(YELLOW)🔄 同步依赖...$(NC)"
	pip install -e . --upgrade
	@echo "$(GREEN)✅ 依赖同步完成$(NC)"

add:  ## 添加依赖包 (用法: make add PKG=package_name)
	@echo "$(YELLOW)📦 添加依赖: $(PKG)...$(NC)"
	pip install $(PKG)

add-dev:  ## 添加开发依赖 (用法: make add-dev PKG=package_name)
	@echo "$(YELLOW)📦 添加开发依赖: $(PKG)...$(NC)"
	pip install $(PKG)

remove:  ## 移除依赖包 (用法: make remove PKG=package_name)
	@echo "$(YELLOW)🗑️  移除依赖: $(PKG)...$(NC)"
	pip uninstall $(PKG)

# ============================================================
# 代码质量
# ============================================================

format:  ## 格式化代码
	@echo "$(YELLOW)🎨 格式化代码...$(NC)"
	python3 -m ruff check --select I --fix app/ 2>/dev/null || true
	python3 -m ruff format app/ 2>/dev/null || python3 -m black app/
	@echo "$(GREEN)✅ 格式化完成$(NC)"

lint:  ## 代码检查
	@echo "$(YELLOW)🔍 代码检查...$(NC)"
	python3 -m ruff check app/ 2>/dev/null || python3 -m flake8 app/
	@echo "$(GREEN)✅ 检查完成$(NC)"

fix:  ## 自动修复代码问题
	@echo "$(YELLOW)🔧 自动修复代码问题...$(NC)"
	python3 -m ruff check --fix app/ 2>/dev/null || true
	python3 -m ruff format app/ 2>/dev/null || python3 -m black app/
	@echo "$(GREEN)✅ 修复完成$(NC)"

type-check:  ## 类型检查
	@echo "$(YELLOW)🔍 类型检查...$(NC)"
	python3 -m mypy app/ --ignore-missing-imports
	@echo "$(GREEN)✅ 类型检查完成$(NC)"

security:  ## 安全检查
	@echo "$(YELLOW)🔒 安全检查...$(NC)"
	python3 -m bandit -r app/ -ll
	@echo "$(GREEN)✅ 安全检查完成$(NC)"

test:  ## 运行测试
	@echo "$(YELLOW)🧪 运行测试...$(NC)"
	python3 -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-quick:  ## 快速测试
	@echo "$(YELLOW)⚡ 快速测试...$(NC)"
	python3 -m pytest tests/ -v

# ============================================================
# RAG 评估
# ============================================================

EVAL_RAG_MODE    ?= basic
EVAL_OUTPUT_DIR  ?= reports
V4_JUDGE_MODEL   ?= qwen3.5-plus
V4_SECTION_PRIOR ?= 0.10

V4_EVAL_ENV = RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=rewrite \
	RERANKER_TYPE=cross_encoder RERANK_COARSE_TOP_K=20 \
	EVAL_JUDGE_MODEL=$(V4_JUDGE_MODEL)
V4_EVAL_ARGS = --split dev --retrieval-metrics minimal --ragas-max-workers 4

eval-rag:  ## 运行 RAG 检索评估（用法: make eval-rag [EVAL_RAG_MODE=basic|enhanced]）
	@echo "$(YELLOW)🧪 运行 RAG 检索评估 (mode=$(EVAL_RAG_MODE))...$(NC)"
	RAG_MODE=$(EVAL_RAG_MODE) .venv/bin/python -m tests.evaluation.evaluate_rag --output $(EVAL_OUTPUT_DIR)/eval_$(EVAL_RAG_MODE)_latest.json

index-rag-v4-basic:  ## 用审计后的 SOP 重建 Basic 基线索引
	.venv/bin/python -m tests.evaluation.index_rag_variant --basic --strategy legacy

index-rag-v4-raw:  ## 构建 v4 原始分块隔离索引
	.venv/bin/python -m tests.evaluation.index_rag_variant \
		--collection biz_enhanced_v4_raw --strategy legacy

index-rag-v4-prefix:  ## 构建 v4 标题/section 前缀隔离索引
	.venv/bin/python -m tests.evaluation.index_rag_variant \
		--collection biz_enhanced_v4_prefix --strategy legacy --include-section-prefix

index-rag-v4-child:  ## 构建 v4 H3 child 隔离索引（也供 parent context 使用）
	.venv/bin/python -m tests.evaluation.index_rag_variant \
		--collection biz_enhanced_v4_child --strategy section_child --include-section-prefix

eval-rag-v4-clean-cache:  ## 清理 v4 专用 Judge 缓存（不影响常规缓存）
	rm -f $(EVAL_OUTPUT_DIR)/judge_cache_v4_*.sqlite

eval-rag-v4-basic:  ## 使用 v1.4.0 数据和独立 Judge 缓存重跑 Basic
	RAG_MODE=basic EVAL_JUDGE_MODEL=$(V4_JUDGE_MODEL) \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_basic.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_basic_dev_v4.json

eval-rag-v4-raw:  ## v4 消融 1：原始分块
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_raw \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_raw.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_raw.json

eval-rag-v4-prefix:  ## v4 消融 2：标题和 section 前缀
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_prefix \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_prefix.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_prefix.json

eval-rag-v4-child:  ## v4 消融 3：H3 child 检索
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_child \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_child.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_child.json

eval-rag-v4-parent:  ## v4 消融 4：child retrieval + H2 parent context
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_child RAG_PARENT_CONTEXT=true \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_parent.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_parent.json

eval-rag-v4-adaptive:  ## v4 自适应路由（V4_SECTION_PRIOR=0.05/0.10/0.15）
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_child \
		RAG_PARENT_CONTEXT=true RAG_QUERY_ROUTING=true RAG_SECTION_PRIOR=$(V4_SECTION_PRIOR) \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_adaptive_$(V4_SECTION_PRIOR).sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag $(V4_EVAL_ARGS) \
		--output $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_adaptive_$(V4_SECTION_PRIOR).json

eval-rag-v4-priors:  ## 运行 section prior 0.05/0.10/0.15 消融
	$(MAKE) eval-rag-v4-adaptive V4_SECTION_PRIOR=0.05
	$(MAKE) eval-rag-v4-adaptive V4_SECTION_PRIOR=0.10
	$(MAKE) eval-rag-v4-adaptive V4_SECTION_PRIOR=0.15

eval-rag-v4-ablation: eval-rag-v4-clean-cache index-rag-v4-basic index-rag-v4-raw index-rag-v4-prefix index-rag-v4-child  ## 完整 v4 单变量消融
	$(MAKE) eval-rag-v4-basic
	$(MAKE) eval-rag-v4-raw
	$(MAKE) eval-rag-v4-prefix
	$(MAKE) eval-rag-v4-child
	$(MAKE) eval-rag-v4-parent
	$(MAKE) eval-rag-v4-priors

eval-rag-v4-ablation-resume: eval-rag-v4-clean-cache index-rag-v4-raw index-rag-v4-prefix index-rag-v4-child  ## Basic 已成功时从 Enhanced 继续
	$(MAKE) eval-rag-v4-basic
	$(MAKE) eval-rag-v4-raw
	$(MAKE) eval-rag-v4-prefix
	$(MAKE) eval-rag-v4-child
	$(MAKE) eval-rag-v4-parent
	$(MAKE) eval-rag-v4-priors

eval-rag-v4-gate:  ## 检查 dev 质量、分类退化和 P95 延迟门禁
	.venv/bin/python -m tests.evaluation.check_rag_v4_gate \
		--baseline $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_raw.json \
		--candidate $(EVAL_OUTPUT_DIR)/eval_enhanced_dev_v4_adaptive_$(V4_SECTION_PRIOR).json

eval-rag-v4-test: eval-rag-v4-gate  ## dev 达标后，仅运行一次冻结 test 与生成评估
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_child \
		RAG_PARENT_CONTEXT=true RAG_QUERY_ROUTING=true RAG_SECTION_PRIOR=$(V4_SECTION_PRIOR) \
		EVAL_JUDGE_CACHE_PATH=$(EVAL_OUTPUT_DIR)/judge_cache_v4_test.sqlite \
		.venv/bin/python -m tests.evaluation.evaluate_rag --split test \
		--retrieval-metrics minimal --ragas-max-workers 4 --with-generation \
		--generation-metrics full --output $(EVAL_OUTPUT_DIR)/eval_enhanced_test_v4_final.json
	$(V4_EVAL_ENV) ENHANCED_COLLECTION_NAME=biz_enhanced_v4_child \
		RAG_PARENT_CONTEXT=true RAG_QUERY_ROUTING=true RAG_SECTION_PRIOR=$(V4_SECTION_PRIOR) \
		.venv/bin/python -m tests.evaluation.evaluate_generation --split test \
		--output $(EVAL_OUTPUT_DIR)/generation_enhanced_test_v4_final.json
	.venv/bin/python -m tests.evaluation.check_rag_v4_gate --stage final \
		--candidate $(EVAL_OUTPUT_DIR)/eval_enhanced_test_v4_final.json \
		--generation $(EVAL_OUTPUT_DIR)/generation_enhanced_test_v4_final.json

eval-rag-full:  ## 运行完整 RAG 评估（检索 + 生成 + correctness）
	@echo "$(YELLOW)🧪 运行完整 RAG 评估 (mode=$(EVAL_RAG_MODE), 含生成评估)...$(NC)"
	RAG_MODE=$(EVAL_RAG_MODE) .venv/bin/python -m tests.evaluation.evaluate_rag \
		--with-generation --generation-metrics full \
		--output $(EVAL_OUTPUT_DIR)/eval_$(EVAL_RAG_MODE)_full_latest.json

eval-generation:  ## 运行 RAG 生成质量独立评估
	@echo "$(YELLOW)🧪 运行 RAG 生成质量评估 (mode=$(EVAL_RAG_MODE))...$(NC)"
	RAG_MODE=$(EVAL_RAG_MODE) .venv/bin/python -m tests.evaluation.evaluate_generation \
		--output $(EVAL_OUTPUT_DIR)/gen_$(EVAL_RAG_MODE)_latest.json

eval-compare:  ## 对比 basic vs enhanced 模式评估结果
	@echo "$(YELLOW)📊 对比 Basic vs Enhanced 评估结果...$(NC)"
	.venv/bin/python -m tests.evaluation.compare_reports \
		--basic $(EVAL_OUTPUT_DIR)/eval_basic_latest.json \
		--enhanced $(EVAL_OUTPUT_DIR)/eval_enhanced_latest.json \
		--output $(EVAL_OUTPUT_DIR)/comparison_latest.json

eval-ablation:  ## 运行消融实验（跨参数组合）
	@echo "$(YELLOW)🔬 运行消融实验...$(NC)"
	.venv/bin/python -m tests.evaluation.run_ablation --output $(EVAL_OUTPUT_DIR)/ablation_latest.csv

eval-validate-dataset:  ## 验证评估数据集质量
	@echo "$(YELLOW)🔍 验证评估数据集质量...$(NC)"
	.venv/bin/python -m tests.evaluation.validate_dataset

eval-baseline-save:  ## 保存当前评估结果为基线
	@echo "$(YELLOW)📌 保存评估基线 (mode=$(EVAL_RAG_MODE))...$(NC)"
	@mkdir -p $(EVAL_OUTPUT_DIR)
	RAG_MODE=$(EVAL_RAG_MODE) .venv/bin/python -m tests.evaluation.evaluate_rag \
		--with-generation --generation-metrics full \
		--output $(EVAL_OUTPUT_DIR)/baseline_$(EVAL_RAG_MODE).json
	@echo "$(GREEN)✅ 基线已保存: $(EVAL_OUTPUT_DIR)/baseline_$(EVAL_RAG_MODE).json$(NC)"

eval-baseline-check:  ## 与基线对比，检查指标退化
	@echo "$(YELLOW)📊 与基线对比检查 (mode=$(EVAL_RAG_MODE))...$(NC)"
	@if [ ! -f "$(EVAL_OUTPUT_DIR)/baseline_$(EVAL_RAG_MODE).json" ]; then \
		echo "$(RED)❌ 基线文件不存在，请先运行: make eval-baseline-save$(NC)"; \
		exit 1; \
	fi
	RAG_MODE=$(EVAL_RAG_MODE) .venv/bin/python -m tests.evaluation.evaluate_rag \
		--with-generation --generation-metrics full \
		--output $(EVAL_OUTPUT_DIR)/eval_$(EVAL_RAG_MODE)_check.json
	.venv/bin/python -c "\
import json, sys; \
baseline = json.load(open('$(EVAL_OUTPUT_DIR)/baseline_$(EVAL_RAG_MODE).json')); \
current = json.load(open('$(EVAL_OUTPUT_DIR)/eval_$(EVAL_RAG_MODE)_check.json')); \
ret_b = baseline['retrieval_metrics']; \
ret_c = current['retrieval_metrics']; \
gen_b = baseline.get('generation_metrics') or {}; \
gen_c = current.get('generation_metrics') or {}; \
degraded = []; \
DEGRADE_THRESHOLD = 0.05; \
for m in ['context_precision','context_recall','context_relevancy','context_entity_recall']: \
    if m in ret_b and m in ret_c and ret_c[m] is not None and ret_b[m] is not None: \
        delta = ret_b[m] - ret_c[m]; \
        if delta > DEGRADE_THRESHOLD: \
            degraded.append(f'{m}: {ret_b[m]:.3f}→{ret_c[m]:.3f} (↓{delta:.3f})'); \
for m in ['faithfulness','answer_relevancy','answer_correctness']: \
    if m in gen_b and m in gen_c and gen_c[m] is not None and gen_b[m] is not None: \
        delta = gen_b[m] - gen_c[m]; \
        if delta > DEGRADE_THRESHOLD: \
            degraded.append(f'{m}: {gen_b[m]:.3f}→{gen_c[m]:.3f} (↓{delta:.3f})'); \
if degraded: \
    print(f'⚠️  发现 {len(degraded)} 个指标退化（>{DEGRADE_THRESHOLD}）:'); \
    for d in degraded: print(f'  - {d}'); \
    sys.exit(1); \
else: \
    print('✅ 所有指标均未退化（退化阈值={DEGRADE_THRESHOLD}）'); \
"
	@echo "$(GREEN)✅ 基线对比检查完成$(NC)"

check-all:  ## 运行所有检查
	@echo "$(YELLOW)🚀 运行所有检查...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test
	@echo "$(GREEN)✅ 所有检查通过！$(NC)"

pre-commit-install:  ## 安装 pre-commit hooks
	@echo "$(YELLOW)🔗 安装 pre-commit hooks...$(NC)"
	python3 -m pre_commit install
	python3 -m pre_commit install --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit hooks 安装完成$(NC)"

pre-commit:  ## 运行 pre-commit 检查
	@echo "$(YELLOW)🔍 运行 pre-commit 检查...$(NC)"
	python3 -m pre_commit run --all-files

coverage:  ## 查看测试覆盖率报告
	@echo "$(YELLOW)📊 生成覆盖率报告...$(NC)"
	python3 -m pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ 覆盖率报告已生成: htmlcov/index.html$(NC)"
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "请手动打开 htmlcov/index.html"

# ============================================================
# 其他工具
# ============================================================

clean:  ## 清理临时文件
	@echo "$(YELLOW)🧹 清理临时文件...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	rm -f server.pid server.log
	rm -f mcp_cls.pid mcp_cls.log
	rm -f mcp_monitor.pid mcp_monitor.log
	rm -rf uploads/*.tmp 2>/dev/null || true
	@echo "$(GREEN)✅ 清理完成$(NC)"

shell:  ## 启动 Python shell
	@echo "$(YELLOW)🐍 启动 Python shell...$(NC)"
	python3 -i -c "import sys; sys.path.insert(0, '.'); from app.config import config; print('环境已加载，config 对象可用')"

ipython:  ## 启动 IPython shell
	@echo "$(YELLOW)🐍 启动 IPython shell...$(NC)"
	python3 -m IPython

docs:  ## 打开 API 文档
	@echo "$(YELLOW)📚 API 文档地址: $(SERVER_URL)/docs$(NC)"
	@open $(SERVER_URL)/docs 2>/dev/null || xdg-open $(SERVER_URL)/docs 2>/dev/null || echo "请手动打开 $(SERVER_URL)/docs"

watch:  ## 监视文件变化并自动运行测试
	@echo "$(YELLOW)👀 监视文件变化...$(NC)"
	python3 -m pytest_watch -- -v

logs:  ## 查看服务日志
	@echo "$(YELLOW)📜 查看服务日志...$(NC)"
	@if [ -f server.log ]; then \
		tail -f server.log; \
	else \
		echo "$(RED)日志文件不存在$(NC)"; \
		echo "$(YELLOW)提示: 使用 make start 启动服务后会生成日志$(NC)"; \
	fi

# ============================================================
# Docker 管理（应用容器化）
# ============================================================

DOCKER_IMAGE   ?= oncall-mind
DOCKER_TAG     ?= latest
DOCKER_REPO    ?= blueblue22/oncall-mind
ENV_FILE       ?= .env.docker

docker-build:  ## 构建 Docker 镜像（精简版）
	@echo "$(YELLOW)🐳 构建 Docker 镜像 $(DOCKER_IMAGE):$(DOCKER_TAG) ...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✅ 镜像构建完成: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"
	@docker images $(DOCKER_IMAGE):$(DOCKER_TAG) --format "   大小: {{.Size}}"

docker-up:  ## 一键启动所有服务（基础设施 + 应用）
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)🚀 启动所有 Docker 服务$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	ENV_FILE=$(ENV_FILE) docker compose up -d
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✅ 所有服务已启动！$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "   $(CYAN)FastAPI:$(NC)    http://localhost:9900"
	@echo "   $(CYAN)API Docs:$(NC)   http://localhost:9900/docs"
	@echo "   $(CYAN)Attu UI:$(NC)    http://localhost:8000"
	@echo "   $(CYAN)Redis:$(NC)      localhost:6379"

docker-down:  ## 停止所有 Docker 服务
	@echo "$(YELLOW)🛑 停止所有 Docker 服务...$(NC)"
	docker compose down
	@echo "$(GREEN)✅ 所有服务已停止$(NC)"

docker-logs:  ## 查看应用容器日志
	docker compose logs -f oncall-mind

docker-status:  ## 查看 Docker 服务状态
	@echo "$(YELLOW)🔍 Docker 服务状态:$(NC)"
	@docker compose ps

docker-push:  ## 推送镜像到 Docker Hub
	@echo "$(YELLOW)📤 推送镜像到 Docker Hub...$(NC)"
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REPO):$(DOCKER_TAG)
	docker push $(DOCKER_REPO):$(DOCKER_TAG)
	@echo "$(GREEN)✅ 已推送: $(DOCKER_REPO):$(DOCKER_TAG)$(NC)"

docker-push-versioned:  ## 推送带版本号的镜像
	@echo "$(YELLOW)📤 推送版本化镜像...$(NC)"
	@VERSION=$$(grep 'version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/'); \
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REPO):v$$VERSION; \
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REPO):v$$VERSION-lite; \
	docker push $(DOCKER_REPO):v$$VERSION; \
	docker push $(DOCKER_REPO):v$$VERSION-lite; \
	echo "$(GREEN)✅ 已推送: $(DOCKER_REPO):v$$VERSION 和 v$$VERSION-lite$(NC)"

docker-shell:  ## 进入应用容器 shell
	docker exec -it oncall-mind bash
