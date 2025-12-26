"""Qwen 集成测试 - Parser -> Embedder -> Reranker

完整测试流程：
1. 使用 Parser 解析文档
2. 使用 Chunker 分块
3. 使用 Embedder 生成向量
4. 存储到向量数据库
5. 检索
6. 使用 Qwen Reranker 重排序

环境变量要求：
- QWEN_API_KEY: Qwen API密钥

运行测试：
  export QWEN_API_KEY='your-api-key-here'
  python tests/test_qwen_integration.py
"""

import asyncio
import os
from pathlib import Path
import pytest
from langrag import KnowledgeBaseManager

# 从环境变量读取 API Key
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

# 如果没有 API Key，跳过整个模块
pytestmark = pytest.mark.skipif(
    not QWEN_API_KEY,
    reason="QWEN_API_KEY environment variable not set. Set it to run Qwen integration tests."
)


async def test_full_integration():
    """完整集成测试"""
    
    print("=" * 70)
    print("🚀 Qwen 集成测试：Parser -> Embedder -> Reranker")
    print("=" * 70)
    print()
    
    # ========== 1. 创建知识库管理器 ==========
    print("📦 步骤 1: 初始化知识库管理器")
    print("-" * 70)
    mgr = KnowledgeBaseManager()
    print("✓ 管理器初始化成功\n")
    
    # ========== 2. 创建数据源 ==========
    print("📦 步骤 2: 创建数据源")
    print("-" * 70)
    mgr.create_datasource(
        name="qwen_test_ds",
        store_type="in_memory",
        params={}
    )
    print("✓ 数据源创建成功: qwen_test_ds\n")
    
    # ========== 3. 创建知识库（带 Qwen Reranker）==========
    print("📦 步骤 3: 创建知识库（配置 Qwen Reranker）")
    print("-" * 70)
    mgr.create_knowledge_base(
        kb_id="qwen_kb",
        datasource_names=["qwen_test_ds"],
        embedder_config={
            "type": "mock",
            "params": {"dimension": 384}
        },
        reranker_config={
            "type": "qwen",
            "params": {
                "api_key": QWEN_API_KEY,
                "model": "qwen3-rerank",
                "timeout": 30.0
            }
        }
    )
    print("✓ 知识库创建成功: qwen_kb")
    print("  - Embedder: MockEmbedder (384维)")
    print("  - Reranker: QwenReranker (qwen3-rerank)")
    print()
    
    # ========== 4. 准备测试文档 ==========
    print("📦 步骤 4: 准备测试文档")
    print("-" * 70)
    
    test_docs = {
        "ml_basics.txt": """机器学习基础知识

什么是机器学习？
机器学习是人工智能的一个重要分支。它使计算机能够从数据中学习，
而不需要明确编程每一个决策规则。机器学习算法可以从经验中自动改进性能。

机器学习的主要类型：
1. 监督学习：从标记的训练数据中学习，如分类、回归等
2. 无监督学习：从未标记的数据中发现模式，如聚类、降维等
3. 强化学习：通过与环境交互来学习最优策略

深度学习是什么？
深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的
层次化表示。深度学习在图像识别、语音识别、自然语言处理等领域
取得了突破性进展。

常见的深度学习架构包括：
- 卷积神经网络（CNN）：主要用于图像处理
- 循环神经网络（RNN）：主要用于序列数据
- Transformer：现代 NLP 的基础架构
""",
        
        "weather.txt": """今日天气预报

北京地区天气：
今天白天多云转晴，最高温度 25℃
夜间晴，最低温度 15℃
风力：3-4级
空气质量：良

明日天气预报：
明天全天晴朗，适合出行
最高温度 27℃，最低温度 16℃
""",
        
        "cooking.md": """# 西红柿炒鸡蛋食谱

## 材料准备
- 西红柿 2个
- 鸡蛋 3个
- 葱花 适量
- 盐、糖 适量

## 制作步骤
1. 鸡蛋打散，加少许盐
2. 热油炒鸡蛋，炒至金黄盛出
3. 西红柿切块，炒软
4. 加入炒好的鸡蛋
5. 调味即可
"""
    }
    
    # 创建测试文件
    test_files = []
    for filename, content in test_docs.items():
        filepath = Path(filename)
        filepath.write_text(content, encoding="utf-8")
        test_files.append(filepath)
        print(f"  - 创建文档: {filename} ({len(content)} 字符)")
    
    print()
    
    # ========== 5. 索引文档 ==========
    print("📦 步骤 5: 索引文档")
    print("-" * 70)
    
    total_chunks = 0
    for filepath in test_files:
        try:
            result = mgr.index_document("qwen_kb", str(filepath))
            if result["status"] == "success":
                num_chunks = result["num_chunks"]
                total_chunks += num_chunks
                print(f"  ✓ {filepath.name}: {num_chunks} chunks")
            else:
                print(f"  ✗ {filepath.name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ {filepath.name}: 索引失败 - {e}")
    
    print(f"\n✓ 索引完成，共 {total_chunks} 个 chunks\n")
    
    # ========== 6. 测试检索（不使用 Reranker）==========
    print("📦 步骤 6: 测试检索（不使用 Reranker）")
    print("-" * 70)
    
    query = "深度学习的主要应用领域有哪些？"
    print(f"查询: {query}\n")
    
    # 临时禁用 reranker 进行对比
    kb = mgr.knowledge_bases.get("qwen_kb")
    original_reranker = kb.reranker
    kb.reranker = None
    
    results_no_rerank = await mgr.search_async("qwen_kb", query, top_k=5)
    
    print("原始检索结果（按向量相似度排序）:")
    for i, result in enumerate(results_no_rerank, 1):
        preview = result["content"].replace('\n', ' ')[:80]
        print(f"  {i}. [Score: {result['score']:.4f}] {preview}...")
    print()
    
    # ========== 7. 测试检索（使用 Qwen Reranker）==========
    print("📦 步骤 7: 测试检索（使用 Qwen Reranker 重排序）")
    print("-" * 70)
    
    # 恢复 reranker
    kb.reranker = original_reranker
    
    print(f"查询: {query}\n")
    print("🔄 调用 Qwen Reranker API...")
    
    try:
        results_with_rerank = await mgr.search_async("qwen_kb", query, top_k=5)
        
        print("✓ Qwen Reranker 重排序后的结果:")
        for i, result in enumerate(results_with_rerank, 1):
            preview = result["content"].replace('\n', ' ')[:80]
            print(f"  {i}. [Score: {result['score']:.4f}] {preview}...")
        print()
        
        # ========== 8. 对比分析 ==========
        print("📦 步骤 8: 对比分析")
        print("-" * 70)
        
        print("排序变化:")
        for i in range(min(3, len(results_no_rerank))):
            orig_preview = results_no_rerank[i]["content"].replace('\n', ' ')[:40]
            rerank_preview = results_with_rerank[i]["content"].replace('\n', ' ')[:40]
            
            if orig_preview != rerank_preview:
                print(f"  位置 {i+1}:")
                print(f"    原始: {orig_preview}...")
                print(f"    重排: {rerank_preview}...")
        
        print()
        
    except Exception as e:
        print(f"✗ Qwen Reranker 调用失败: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # ========== 9. 测试更多查询 ==========
    print("📦 步骤 9: 测试不同类型的查询")
    print("-" * 70)
    
    test_queries = [
        "什么是监督学习和无监督学习？",
        "今天天气怎么样？",
        "如何做西红柿炒鸡蛋？"
    ]
    
    for query_text in test_queries:
        print(f"\n查询: {query_text}")
        try:
            results = await mgr.search_async("qwen_kb", query_text, top_k=3)
            print(f"  找到 {len(results)} 个结果:")
            for i, result in enumerate(results, 1):
                preview = result["content"].replace('\n', ' ')[:60]
                print(f"    {i}. [{result['score']:.4f}] {preview}...")
        except Exception as e:
            print(f"  ✗ 查询失败: {e}")
    
    print()
    
    # ========== 清理 ==========
    print("📦 清理测试文件")
    print("-" * 70)
    for filepath in test_files:
        try:
            filepath.unlink()
            print(f"  ✓ 删除: {filepath.name}")
        except Exception as e:
            print(f"  ✗ 删除失败 {filepath.name}: {e}")
    
    print()
    print("=" * 70)
    print("✅ 集成测试完成！")
    print("=" * 70)
    print()
    print("测试总结:")
    print(f"  - 索引文档数: {len(test_files)}")
    print(f"  - 总 chunks: {total_chunks}")
    print(f"  - 测试查询数: {len(test_queries) + 2}")
    print(f"  - Qwen Reranker: {'✓ 正常工作' if results_with_rerank else '✗ 失败'}")


if __name__ == "__main__":
    asyncio.run(test_full_integration())

