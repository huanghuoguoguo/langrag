import json
import logging
import time
from dataclasses import dataclass, field

from langrag.agent.tool import Tool
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When answering questions:
1. If the question requires information from external knowledge bases, use the search_knowledge_base tool.
2. If you can answer directly without external knowledge (greetings, general questions, etc.), respond immediately without using tools.
3. You may call tools multiple times if needed to gather sufficient information.
4. After gathering information, provide a comprehensive answer based on the retrieved context.

Always be helpful, accurate, and concise."""


@dataclass
class ToolCall:
    """Record of a single tool call."""
    name: str
    arguments: dict
    result: str
    error: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class AgentStep:
    """Record of a single agent step (LLM call + tool executions)."""
    step: int
    thought: str | None = None  # LLM's reasoning/content before tool call
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class AgentResult:
    """Complete result of agent execution with trace."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    total_tool_calls: int = 0
    total_elapsed_ms: float = 0.0
    finished_reason: str = "complete"  # "complete" | "max_steps" | "error"


async def run_agent(
    llm: BaseLLM,
    tools: list[Tool],
    messages: list[dict],
    max_steps: int = 5,
    system_prompt: str | None = None,
    return_trace: bool = False,
) -> str | AgentResult:
    """
    Run a simple ReAct/Tool-use loop.

    Args:
        llm: The LLM to use.
        tools: List of available tools.
        messages: Chat history (should NOT include system message, will be prepended).
        max_steps: Maximum number of tool execution steps to prevent loops.
        system_prompt: Custom system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.
        return_trace: If True, returns AgentResult with full trace. If False, returns just the answer string.

    Returns:
        If return_trace=False: Final answer string.
        If return_trace=True: AgentResult with answer and execution trace.
    """
    start_time = time.perf_counter()
    trace_steps: list[AgentStep] = []
    total_tool_calls = 0

    # Convert tools to OpenAI format
    openai_tools = []
    tool_map = {}
    for tool in tools:
        tool_map[tool.name] = tool
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        })

    # Build message list with system prompt
    effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    current_messages = [{"role": "system", "content": effective_system_prompt}]

    # Add user messages (filter out any existing system messages to avoid duplication)
    for msg in messages:
        if msg.get("role") != "system":
            current_messages.append(msg)

    for step_num in range(max_steps):
        step_start = time.perf_counter()
        current_step = AgentStep(step=step_num + 1)

        # 1. Call LLM
        try:
            # Check if llm has chat_dict, otherwise fall back to chat (no tools)
            if hasattr(llm, "chat_dict"):
                response_msg = await _call_llm_async(llm, current_messages, tools=openai_tools)
            else:
                answer = await _call_llm_chat_async(llm, current_messages)
                return _build_result(answer, trace_steps, total_tool_calls, start_time, "complete", return_trace)

        except Exception as e:
            logger.error(f"[Agent] LLM call failed: {e}")
            raise e

        # 2. Check for tool calls
        tool_calls = response_msg.get("tool_calls")
        content = response_msg.get("content")
        current_step.thought = content

        # Always append the assistant's message to history
        current_messages.append(response_msg)

        if not tool_calls:
            # No tool calls - LLM is done, return the answer
            current_step.elapsed_ms = (time.perf_counter() - step_start) * 1000
            trace_steps.append(current_step)
            return _build_result(content or "", trace_steps, total_tool_calls, start_time, "complete", return_trace)

        # 3. Execute tools
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args_str = tool_call["function"]["arguments"]
            call_id = tool_call["id"]

            logger.info(f"[Agent] Tool Call: {func_name}({func_args_str})")
            tool_start = time.perf_counter()
            tool_record = ToolCall(name=func_name, arguments={}, result="", error=None)

            if func_name in tool_map:
                try:
                    args = json.loads(func_args_str)
                    tool_record.arguments = args
                    tool_instance = tool_map[func_name]

                    # Execute
                    result = await tool_instance.func(**args)
                    result_str = str(result)
                    tool_record.result = result_str

                except Exception as e:
                    logger.error(f"[Agent] Tool execution failed: {e}")
                    result_str = f"Error: {str(e)}"
                    tool_record.error = str(e)
                    tool_record.result = result_str
            else:
                result_str = f"Error: Tool {func_name} not found"
                tool_record.error = f"Tool {func_name} not found"
                tool_record.result = result_str

            tool_record.elapsed_ms = (time.perf_counter() - tool_start) * 1000
            current_step.tool_calls.append(tool_record)
            total_tool_calls += 1

            # 4. Append tool result
            current_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": result_str
            })

        current_step.elapsed_ms = (time.perf_counter() - step_start) * 1000
        trace_steps.append(current_step)

    # Reached max steps
    return _build_result(
        "Agent reached maximum steps without final answer.",
        trace_steps, total_tool_calls, start_time, "max_steps", return_trace
    )


def _build_result(
    answer: str,
    steps: list[AgentStep],
    total_tool_calls: int,
    start_time: float,
    reason: str,
    return_trace: bool
) -> str | AgentResult:
    """Helper to build the return value."""
    if not return_trace:
        return answer

    return AgentResult(
        answer=answer,
        steps=steps,
        total_steps=len(steps),
        total_tool_calls=total_tool_calls,
        total_elapsed_ms=(time.perf_counter() - start_time) * 1000,
        finished_reason=reason
    )

async def _call_llm_async(llm, messages, **kwargs):
    import asyncio
    return await asyncio.to_thread(llm.chat_dict, messages, **kwargs)

async def _call_llm_chat_async(llm, messages):
    import asyncio
    return await asyncio.to_thread(llm.chat, messages)
