"""异步辅助工具 - 处理同步/异步上下文转换

本模块提供工具函数来处理在同步代码中调用异步函数的场景。
主要用于向后兼容和在不支持异步的环境中运行异步代码。

Examples:
    >>> async def fetch_data():
    ...     return "data"
    >>>
    >>> # 在同步代码中调用
    >>> result = run_async_in_sync_context(fetch_data())
    >>> print(result)
    data
"""

import asyncio
import concurrent.futures
from typing import TypeVar, Coroutine
from loguru import logger

T = TypeVar('T')


def run_async_in_sync_context(coro: Coroutine[None, None, T]) -> T:
    """在同步上下文中运行异步协程

    自动处理：
    1. 检测现有事件循环
    2. 尝试使用 nest_asyncio（Jupyter 等环境）
    3. 回退到线程池执行
    4. 无循环时使用 asyncio.run()

    Warning:
        在已有事件循环的环境中（Jupyter, FastAPI）可能导致性能下降。
        建议直接使用异步版本的 API。

    Args:
        coro: 要执行的协程对象

    Returns:
        协程的返回值

    Examples:
        >>> async def get_user(user_id: int):
        ...     # 模拟异步数据库查询
        ...     await asyncio.sleep(0.1)
        ...     return {"id": user_id, "name": "Alice"}
        >>>
        >>> # 在同步代码中使用
        >>> user = run_async_in_sync_context(get_user(1))
        >>> print(user["name"])
        Alice
    """
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()

        # 在运行的事件循环中 - 需要特殊处理
        logger.debug(
            "Detected running event loop. Consider using async methods directly "
            "for better performance."
        )

        # 尝试使用 nest_asyncio 处理嵌套循环
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)

        except ImportError:
            # nest_asyncio 未安装，使用线程池回退
            logger.debug("nest_asyncio not available, using thread pool fallback")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()

    except RuntimeError:
        # 没有运行的事件循环 - 直接使用 asyncio.run()
        return asyncio.run(coro)
