"""
示例LLM实现 (Example LLM Implementations)

这些实现可以直接使用，也可以作为参考自行实现BaseLLM接口。
外部应用可以注入自己的实现来替换这些默认实现。

Example Implementations:

- OpenAILLM: OpenAI兼容API的实现
- LocalLLM: 本地模型(llama.cpp)的实现

外部应用可以选择：
1. 直接使用这些实现
2. 基于这些实现进行定制
3. 完全实现自己的BaseLLM子类
"""

from .local import LocalLLM
from .openai import OpenAILLM

__all__ = ["LocalLLM", "OpenAILLM"]
