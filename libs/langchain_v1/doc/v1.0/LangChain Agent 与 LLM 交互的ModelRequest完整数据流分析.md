# LangChain Agent 与 LLM 交互的ModelRequest完整数据流分析

## 引言

在现代AI应用开发中，LangChain 作为最流行的框架之一，为构建基于大语言模型（LLM）的应用提供了强大的工具集。其中 Agent（智能代理）是 LangChain 的核心组件之一，能够让 LLM 具备工具调用、记忆和推理能力。

然而，对于开发者来说，理解 Agent 每次与 LLM 交互时究竟发送了什么内容至关重要。本文将深入剖析 LangChain Agent 的内部机制，结合源码详细解释数据流的完整过程。

## LangChain Agent 的工作原理

### Agent 工厂模式

Agent 的创建始于 `create_agent` 函数。让我们查看这个核心函数：

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]:
```

这个函数构建了一个基于 StateGraph 的有状态图，其中包含多个节点来处理 agent 的执行流程。

### 核心执行节点

Agent 的核心是 `model_node` 函数，它负责每次与 LLM 的交互：

```python
def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
    request = ModelRequest(
        model=model,
        tools=default_tools,
        system_message=system_message,
        response_format=initial_response_format,
        messages=state["messages"],  # 对话历史
        tool_choice=None,
        state=state,
        runtime=runtime,
    )

    if wrap_model_call_handler is None:
        response = _execute_model_sync(request)
    else:
        response = wrap_model_call_handler(request, _execute_model_sync)

    state_updates = {"messages": response.result}
    if response.structured_response is not None:
        state_updates["structured_response"] = response.structured_response

    return state_updates
```

## ModelRequest 数据结构详解

### ModelRequest 的完整定义

`ModelRequest` 是发送给模型的核心数据结构：

```python
@dataclass(init=False)
class ModelRequest:
    """Model request information for the agent."""

    model: BaseChatModel                    # 模型实例
    messages: list[AnyMessage]             # 消息列表（不含系统消息）
    system_message: SystemMessage | None   # 系统消息
    tool_choice: Any | None                # 工具选择策略
    tools: list[BaseTool | dict]           # 可用工具列表
    response_format: ResponseFormat | None # 响应格式
    state: AgentState                      # Agent 状态
    runtime: Runtime[ContextT]             # 运行时上下文
    model_settings: dict[str, Any] = field(default_factory=dict)  # 模型额外设置
```

### 消息拼合过程

在 `_execute_model_sync` 函数中，消息被拼合成最终发送给模型的格式：

```python
def _execute_model_sync(request: ModelRequest) -> ModelResponse:
    # 获取绑定后的模型和响应格式
    model_, effective_response_format = _get_bound_model(request)

    # 🎯 核心：拼合消息列表
    messages = request.messages
    if request.system_message:
        messages = [request.system_message, *messages]  # 系统消息放最前面

    output = model_.invoke(messages)  # 发送给模型

    # 处理模型输出
    handled_output = _handle_model_output(output, effective_response_format)
    messages_list = handled_output["messages"]
    structured_response = handled_output.get("structured_response")

    return ModelResponse(
        result=messages_list,
        structured_response=structured_response,
    )
```

## 最终发送格式：字典 vs 字符串

### 常见误解澄清

很多人误以为 LangChain 发送给模型的是字符串，但实际发送的是**结构化字典格式**。

### BaseChatModel 的处理流程

`BaseChatModel.invoke()` 方法接收多种输入格式：

```python
def invoke(
    self,
    input: LanguageModelInput,  # 可以是 str | list[BaseMessage] | PromptValue
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage:
    config = ensure_config(config)
    return cast(
        "AIMessage",
        cast(
            "ChatGeneration",
            self.generate_prompt(
                [self._convert_input(input)],  # 转换输入
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            ).generations[0][0],
        ).message,
    )
```

### 消息转换过程

`_convert_input` 方法将输入转换为 `ChatPromptValue`：

```python
def _convert_input(self, model_input: LanguageModelInput) -> PromptValue:
    if isinstance(model_input, PromptValue):
        return model_input
    if isinstance(model_input, str):
        return StringPromptValue(text=model_input)
    if isinstance(model_input, Sequence):
        return ChatPromptValue(messages=convert_to_messages(model_input))
    msg = (
        f"Invalid input type {type(model_input)}. "
        "Must be a PromptValue, str, or list of BaseMessages."
    )
    raise ValueError(msg)
```

### 具体模型的格式化

以 OpenAI 为例，`_get_request_payload` 方法将消息转换为 API 期望的字典格式：

```python
def _get_request_payload(
    self,
    input_: LanguageModelInput,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> dict:
    messages = self._convert_input(input_).to_messages()

    payload = {**self._default_params, **kwargs}

    payload["messages"] = [
        _convert_message_to_dict(_convert_from_v1_to_chat_completions(m))
        if isinstance(m, AIMessage)
        else _convert_message_to_dict(m)
        for m in messages
    ]
    return payload
```

### 消息字典转换

`_convert_message_to_dict` 函数将每个 `BaseMessage` 转换为 OpenAI API 格式：

```python
def _convert_message_to_dict(
    message: BaseMessage,
    api: Literal["chat/completions", "responses"] = "chat/completions",
) -> dict:
    """Convert a LangChain message to dictionary format expected by OpenAI."""
    message_dict: dict[str, Any] = {
        "content": _format_message_content(message.content, api=api, role=message.type)
    }

    # 根据消息类型设置 role
    if isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        # 处理工具调用...
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"

    return message_dict
```

### 最终 API 格式

发送给 OpenAI API 的 payload 如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个天气助手"
    },
    {
      "role": "user",
      "content": "帮我查询天气"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"北京\"}"
        }
      }]
    },
    {
      "role": "tool",
      "content": "北京今天晴天，温度25°C",
      "tool_call_id": "call_123"
    }
  ]
}
```

## 如何获取发送给 LLM 的真实字符串

### 方法1：使用 get_buffer_string 函数

LangChain 提供了 `get_buffer_string` 函数来将消息列表转换为字符串：

```python
from langchain_core.messages import get_buffer_string

def get_full_prompt_string(request: ModelRequest) -> str:
    """获取发送给LLM的完整字符串表示"""
    all_messages = []
    if request.system_message:
        all_messages.append(request.system_message)
    all_messages.extend(request.messages)

    return get_buffer_string(all_messages)
```

`get_buffer_string` 的实现：

```python
def get_buffer_string(
    messages: Sequence[BaseMessage],
    human_prefix: str = "Human",
    ai_prefix: str = "AI"
) -> str:
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        else:
            role = m.type

        message = f"{role}: {m.text}"
        string_messages.append(message)

    return "\n".join(string_messages)
```

### 方法2：自定义中间件（推荐）

创建自定义中间件来记录发送给 LLM 的内容：

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import get_buffer_string
from typing import Any

class LoggingMiddleware(AgentMiddleware[AgentState, Any]):
    """记录发送给LLM内容的中间件"""

    def wrap_model_call(self, request, handler):
        """在模型调用前记录完整消息内容"""
        # 获取所有消息
        all_messages = []
        if request.system_message:
            all_messages.append(request.system_message)
        all_messages.extend(request.messages)

        # 转换为字符串格式
        prompt_string = get_buffer_string(all_messages)

        # 记录日志
        print("=== 发送给LLM的完整内容 ===")
        print(prompt_string)
        print("=" * 50)

        # 调用原始handler
        return handler(request)

# 使用中间件
agent = create_agent(
    model="openai:gpt-4",
    tools=[...],
    middleware=[LoggingMiddleware()]
)
```

### 方法3：使用回调机制

通过回调机制拦截消息：

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import get_buffer_string

class MessageLoggingCallback(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None,
        tags=None, metadata=None, **kwargs
    ) -> None:
        """Called when a chat model starts a run."""
        full_prompt = get_buffer_string(messages)
        print(f"发送给LLM的完整内容:\n{full_prompt}")

# 使用回调
agent = create_agent(
    model="openai:gpt-4",
    tools=[...],
    callbacks=[MessageLoggingCallback()]
)
```

## 实际输出示例

使用上述方法，您会看到类似这样的输出：

```
=== 发送给LLM的完整内容 ===
System: 你是一个天气助手
Human: 帮我查询天气
Assistant:
Tool: 北京今天晴天，温度25°C
==================================================
```

## 数据流总结

1. **Agent 创建**：`create_agent()` 构建 StateGraph
2. **消息收集**：从 `state["messages"]` 获取对话历史
3. **Request 构建**：创建 `ModelRequest` 对象
4. **消息拼合**：系统消息 + 对话历史
5. **格式转换**：BaseMessage → 字典格式
6. **模型调用**：发送结构化 payload
7. **结果处理**：解析响应并更新状态

## 关键洞察

- **LangChain 发送的是字典，不是字符串**：尽管有 `get_buffer_string` 用于调试，但实际 API 调用使用结构化格式
- **消息历史至关重要**：每次交互都包含完整的对话上下文
- **中间件是最佳观测点**：通过 `wrap_model_call` 可以精确控制和记录模型交互
- **多层转换**：BaseMessage → ChatPromptValue → API 字典格式

