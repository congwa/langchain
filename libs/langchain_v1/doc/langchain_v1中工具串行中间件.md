# angchain_v1中工具串行中间件

源码中，当模型一次返回 多个 tool_calls 时，会把“还没执行过的 tool call”筛出来，然后为每个 tool call 单独发一个 Send("tools", ...) 交给 ToolNode 执行（这就是“fan-out”，语义上是并行/多路调度）。

如果想要让工具串行执行怎么办

在 langchain_v1 的 create_agent() 里，模型一次返回多个 tool_calls 会 fan-out 成多个 Send("tools", ...)。要做到“不丢弃、全部执行，但一个一个串行跑”，最稳的是在 tool 执行层加全局锁：即使图里同时调度了多个 tool call，也会被锁强制排队。

```
# middleware：把所有工具调用串行化（同步/异步都覆盖）
import asyncio
import threading
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.typing import ContextT


class SequentialToolExecutionMiddleware(AgentMiddleware[AgentState, ContextT]):
    """让同一轮返回的多个 tool_calls 也按顺序一个个执行（不丢弃）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        with self._lock:
            return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        async with self._alock:
            return await handler(request)


# 用法：
# agent = create_agent(model=..., tools=[...], middleware=[SequentialToolExecutionMiddleware()])
# 一次吐 5 个工具，也会 依次执行 1→2→3→4→5
```
