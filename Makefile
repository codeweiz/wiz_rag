.PHONY: setup install dev clean lint test run worker ngrok chroma

# 默认目标
all: setup

# 设置项目
setup: install

# 安装依赖
install:
	uv pip install -e .

# 安装开发依赖
dev:
	uv pip install -e ".[dev]"

# 安装 git 依赖
git:
	uv pip install git+https://github.com/openai/CLIP.git

# 清理项目
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 代码检查
lint:
	uv run ruff check .
	uv run black --check .
	uv run isort --check .
	uv run mypy .

# 运行测试
test:
	uv run pytest

# 运行 Web 服务
run:
	uv run uvicorn pulse_guard.main:app --host 0.0.0.0 --port 8000 --reload

# 运行 Celery Worker
worker:
	uv run celery -A pulse_guard.worker.celery_app worker --loglevel=info

# 启动 ngrok 反向代理
ngrok:
	ngrok http 8000

# 启动 Chromadb
chroma:
	chroma run --path ./data
