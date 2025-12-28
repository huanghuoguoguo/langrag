#!/bin/bash
# LangRAG Web Application 启动脚本

cd "$(dirname "$0")/.."
uv run python -m web.app
