import json
import logging
from typing import Any, Awaitable

from langrag.llm.base import BaseLLM
from langrag.agent.tool import Tool

logger = logging.getLogger(__name__)

async def run_agent(
    llm: BaseLLM,
    tools: list[Tool],
    messages: list[dict],
    max_steps: int = 5
) -> str:
    """
    Run a simple ReAct/Tool-use loop.
    
    Args:
        llm: The LLM to use.
        tools: List of available tools.
        messages: Chat history.
        max_steps: Maximum number of tool execution steps to prevent loops.
        
    Returns:
        Final answer string.
    """
    
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

    current_messages = messages.copy()
    
    for step in range(max_steps):
        # 1. Call LLM
        try:
            # Check if llm has chat_dict, otherwise fall back to chat (no tools)
            if hasattr(llm, "chat_dict"):
                response_msg = await _call_llm_async(llm, current_messages, tools=openai_tools)
            else:
                return await _call_llm_chat_async(llm, current_messages)
                
        except Exception as e:
            logger.error(f"[Agent] LLM call failed: {e}")
            raise e

        # 2. Check for tool calls
        tool_calls = response_msg.get("tool_calls")
        content = response_msg.get("content")
        
        # Always append the assistant's message to history
        current_messages.append(response_msg)

        if not tool_calls:
            # specific case: sometimes LLM puts tool call in content as text (if not fine-tuned), 
            # but we assume proper function calling support here.
            return content or ""

        # 3. Execute tools
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args_str = tool_call["function"]["arguments"]
            call_id = tool_call["id"]
            
            logger.info(f"[Agent] Tool Call: {func_name}({func_args_str})")
            
            if func_name in tool_map:
                try:
                    args = json.loads(func_args_str)
                    tool_instance = tool_map[func_name]
                    
                    # Execute
                    result = await tool_instance.func(**args)
                    result_str = str(result)
                    
                except Exception as e:
                    logger.error(f"[Agent] Tool execution failed: {e}")
                    result_str = f"Error: {str(e)}"
            else:
                result_str = f"Error: Tool {func_name} not found"

            # 4. Append tool result
            current_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": result_str
            })

    return "Agent reached maximum steps without final answer."

async def _call_llm_async(llm, messages, **kwargs):
    import asyncio
    return await asyncio.to_thread(llm.chat_dict, messages, **kwargs)

async def _call_llm_chat_async(llm, messages):
    import asyncio
    return await asyncio.to_thread(llm.chat, messages)
