#!/bin/bash

# 下载 Qwen GGUF 模型脚本
# 用于 LangRAG 本地 LLM 测试

MODEL_DIR="./models"

echo "LangRAG 本地模型下载脚本"
echo "=========================="
echo ""

# 选项1: 小型测试模型 (推荐用于开发测试)
echo "选项1: 下载小型测试模型 (Qwen 0.5B, ~300MB)"
echo "这适合快速测试 LangRAG 的本地 LLM 功能"
echo ""

# 选项2: 完整模型 (7B)
echo "选项2: 下载完整模型 (Qwen 7B, ~4GB)"
echo "这提供更好的回答质量，但下载和运行都需要更多资源"
echo ""

# 让用户选择
read -p "请选择要下载的模型 (1=小型测试模型, 2=完整模型): " choice

case $choice in
    1)
        MODEL_NAME="qwen2-0_5b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf"
        MODEL_SIZE="~300MB"
        ;;
    2)
        MODEL_NAME="qwen2.5-7b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        MODEL_SIZE="~4GB"
        ;;
    *)
        echo "无效选择，使用默认的小型测试模型"
        MODEL_NAME="qwen2-0_5b-instruct-q4_k_m.gguf"
        MODEL_URL="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf"
        MODEL_SIZE="~300MB"
        ;;
esac

echo ""
echo "创建模型目录..."
mkdir -p "$MODEL_DIR"

echo "下载模型文件..."
echo "模型: $MODEL_NAME"
echo "大小: $MODEL_SIZE"
echo "从: $MODEL_URL"
echo "到: $MODEL_DIR/$MODEL_NAME"
echo ""

# 检查文件是否已存在
if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "文件已存在，跳过下载"
else
    echo "开始下载..."
    # 使用 wget 或 curl 下载
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
    else
        echo "错误：需要安装 wget 或 curl 来下载模型"
        echo "或者手动下载模型文件到: $MODEL_DIR/$MODEL_NAME"
        exit 1
    fi
fi

echo ""
echo "模型下载完成！"
echo "路径: $MODEL_DIR/$MODEL_NAME"
echo "大小: $(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)"
echo ""
echo "现在你可以在 LangRAG 中选择 'Local Model (GGUF)' 来使用这个模型。"