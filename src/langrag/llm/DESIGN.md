# LLM Layer Design

## 1. 职责 (Responsibilities)
封装对大语言模型的调用接口。
在 RAG 中，LLM 主要用于：
1. **Embedding**: 生成向量 (Text Embedding)。
2. **Generation**: 生成最终回复 (Chat Completion)。
3. **Routing**: 意图识别 (Function Calling / Reasoning)。

## 2. 接口设计

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
        
    @abstractmethod
    def embedding(self, texts: list[str]) -> list[list[float]]:
        pass
```

*注意：在 Dify 中，`ModelManager` 统一管理了所有模型。在 LangRAG 中，我们可以简化为一个 Adapter。*

## 3. Dify 对应关系
- **Dify Path**: `api/core/model_manager.py`
- **Dify Path**: `api/core/model_runtime/`
