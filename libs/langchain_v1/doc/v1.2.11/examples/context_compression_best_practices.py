# -*- coding: utf-8 -*-
"""
上下文压缩：最佳实践代码示例（含 OpenAI compaction 与「非 OpenAI 专属」方案）

建议放置路径（与配套 Markdown 一致）
------------------------------------
- LangChain 单仓：``langchain/libs/langchain_v1/doc/v1.2.11/examples/context_compression_best_practices.py``
  （同目录上一级为 ``openai-compaction-langchain-guide.md``）
- 其他仓库若仅作内部分享：可置于 ``docs/examples/context_compression_best_practices.py``，
  与 ``docs/openai-compaction-langchain-guide.md`` 并列时，文档内相对链接为 ``./examples/...``。

运行说明
--------
- 本文件以「教学注释」为主；默认不执行网络请求。
- 若需真跑：安装对应 SDK（openai、langchain-openai 等），设置环境变量 OPENAI_API_KEY 等，
  并取消各示例函数末尾的注释或自行写 main。

方案速览
--------
| 方案 | 依赖 OpenAI？ | 核心机制 | 典型场景 |
|------|---------------|----------|----------|
| openai_compaction_with_previous_response_id | 是 | Responses + compaction + 链式 id | 长对话、愿托管状态在 OpenAI 侧 |
| openai_compaction_with_input_array_and_trim | 是 | Responses + compaction + 自研剪枝 | 要自管 items、优化 payload 体积 |
| langchain_openai_compaction_and_chain | 是 | ChatOpenAI 透传 compaction + 链式 | 已在 LangChain 生态内 |
| sliding_window_messages | 否 | 只保留最近 K 条 | 任意模型、实现极简 |
| llm_summarize_history_generic | 否* | 任意 chat 模型生成摘要 | 要多供应商、可读摘要、强可控 |
| summarization_middleware_pattern | 否* | LangChain 中间件 | Agent + 自动触发摘要 |

*摘要模型可选用 Anthropic / Google / 本地等，不强制 OpenAI。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

# =============================================================================
# 方案一：OpenAI — Responses API + 服务端 compaction + previous_response_id 链式
# =============================================================================
#
# 【优点】
#   - 单轮 HTTP payload 小：每轮 largely 只发新 user + 上一响应 id。
#   - 由 OpenAI 服务端 compaction 在超阈值时压缩状态；无需自写摘要 prompt。
#   - 与官方文档「不要手动 prune」一致，心智负担相对较低。
#
# 【缺点】
#   - 强绑定 OpenAI Responses 与当前模型/SDK 能力；换供应商需重写串联逻辑。
#   - 会话连续性依赖 response id 持久化；重试、并发、故障恢复要设计幂等与对账。
#   - compaction 产出 opaque，无法当人类可读审计摘要；合规需单独论证 store/ZDR。
#
# 【适用】长会话、主要用 OpenAI、能接受状态链式托管在平台侧。
# =============================================================================


def openai_compaction_with_previous_response_id() -> None:
    """概念示例：OpenAI 官方 Python SDK（需 openai>=支持 Responses 的版本）。"""
    # from openai import OpenAI
    #
    # client = OpenAI()
    # model = "gpt-5.2"  # 替换为账号内支持 compaction 的模型
    # previous_response_id: str | None = None
    #
    # while True:
    #     user_text = input("User: ").strip()
    #     if not user_text:
    #         break
    #
    #     user_item = {"type": "message", "role": "user", "content": user_text}
    #
    #     kwargs: dict[str, Any] = {
    #         "model": model,
    #         "input": [user_item],
    #         "store": False,  # 合规向：是否存储以法务为准
    #         "context_management": [
    #             {"type": "compaction", "compact_threshold": 50_000}
    #         ],
    #     }
    #     if previous_response_id is not None:
    #         kwargs["previous_response_id"] = previous_response_id
    #
    #     response = client.responses.create(**kwargs)
    #     previous_response_id = response.id
    #
    #     # 将输出展示给用户；若需全量审计请在业务层单独记日志（勿依赖 compaction 明文）
    #     print(response.output_text)  # 属性名以当前 SDK 为准
    pass


# =============================================================================
# 方案二：OpenAI — input 项数组 + compaction + 「最后一个 compaction 之后」剪枝（Latency Tip）
# =============================================================================
#
# 【优点】
#   - 会话全量 items 可落在己方 DB，审计/回放路径清晰（若你持久化 conversation）。
#   - 剪枝后可显著降低下一轮请求 JSON 体积与长尾延迟（官方 latency tip）。
#
# 【缺点】
#   - 剪枝实现错误会导致工具链/状态断裂；必须单测覆盖 tool、reasoning 等顺序。
#   - 与 previous_response_id 模式的「禁止手动 prune」不同，团队文档要写清楚避免混用。
#   - 仍需 OpenAI；维护 items schema 随 API 演进升级。
#
# 【适用】要强控 payload、愿意维护 items 数组与裁剪逻辑的团队。
# =============================================================================


def trim_after_last_compaction(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留最后一个 type==compaction 及其之后的项；若无 compaction 则原样返回。"""
    last_idx: int | None = None
    for i, it in enumerate(items):
        if it.get("type") == "compaction":
            last_idx = i
    if last_idx is None:
        return items
    return items[last_idx:]


def openai_compaction_with_input_array_and_trim() -> None:
    """概念示例：手写 conversation 列表 + extend(output) + 可选剪枝。"""
    # from openai import OpenAI
    #
    # client = OpenAI()
    # model = "gpt-5.2"
    # conversation: list[dict[str, Any]] = [
    #     {"type": "message", "role": "user", "content": "你好"},
    # ]
    #
    # while True:
    #     resp = client.responses.create(
    #         model=model,
    #         input=conversation,
    #         store=False,
    #         context_management=[
    #             {"type": "compaction", "compact_threshold": 50_000}
    #         ],
    #     )
    #     conversation.extend(resp.output)
    #     # 可选：按官方 latency tip 减小下一请求体积（须验证业务正确性）
    #     conversation = trim_after_last_compaction(conversation)
    #
    #     user_text = input("User: ").strip()
    #     if not user_text:
    #         break
    #     conversation.append(
    #         {"type": "message", "role": "user", "content": user_text}
    #     )
    pass


# =============================================================================
# 方案三：OpenAI + LangChain — ChatOpenAI 同时开启 compaction 与 previous_response_id
# =============================================================================
#
# 【优点】
#   - 与现有 LangChain / LangGraph 消息列表集成；链式 id 由库封装 payload 切片。
#   - 仍享受 OpenAI 服务端 compaction；调参集中在 ChatOpenAI 构造参数。
#
# 【缺点】
#   - 依赖 langchain-openai 版本与行为；升级需回归。
#   - 并非「按 compaction 项剪枝」，而是「按 response id 剪」；语义见主文档第 8 节。
#   - 仍绑定 OpenAI Responses 能力。
#
# 【适用】已在 LC 栈内、主模型为 OpenAI、希望控制 HTTP payload 膨胀。
# =============================================================================


def langchain_openai_compaction_and_chain() -> None:
    """概念示例：LangChain ChatOpenAI。"""
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import HumanMessage, AIMessage
    #
    # model = ChatOpenAI(
    #     model="gpt-5.2",
    #     use_previous_response_id=True,
    #     context_management=[{"type": "compaction", "compact_threshold": 50_000}],
    #     store=False,
    # )
    #
    # messages = [HumanMessage("你好")]
    # ai: AIMessage = model.invoke(messages)
    # messages.append(ai)
    # messages.append(HumanMessage("继续问第二个问题"))
    # ai2 = model.invoke(messages)
    # print(ai2.content)
    pass


# =============================================================================
# 方案四：不依赖 OpenAI compaction — 滑动窗口（任意供应商 / 任意 Chat API）
# =============================================================================
#
# 【优点】
#   - 实现极简、无供应商锁定；不调用「压缩 API」、无 compaction pass 费用。
#   - 延迟与行为完全可预测（固定保留 K 条）。
#
# 【缺点】
#   - 粗暴截断：早期用户约束、工具调用链、长期目标可能丢失。
#   - 若 K 太小，工具 call / tool_result 可能不成对，导致模型报错或胡编。
#   - 不是「智能压缩」，只是「丢弃」。
#
# 【适用】短会话、对历史不敏感、或已有外部状态机（DB）补全上下文的场景。
# =============================================================================


def sliding_window_messages(
    messages: list[Any],
    *,
    max_messages: int,
) -> list[Any]:
    """
    仅保留列表末尾 max_messages 条。

    messages: 建议使用 LangChain BaseMessage 列表；任意对象列表亦可。
    max_messages: 建议为偶数且在带 tool 时保证 tool 对与 user/assistant 边界完整。
    """
    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


# =============================================================================
# 方案五：不依赖 OpenAI compaction — 通用「LLM 摘要」压缩（多供应商）
# =============================================================================
#
# 【优点】
#   - 摘要模型可选用 Anthropic、Google、Azure、本地 Ollama 等，**不依赖 OpenAI compaction**。
#   - 产出人类可读摘要，便于日志、侧栏展示、人工抽检（需产品/合规同意）。
#   - 成本可调：小模型摘要 + 大模型主任务。
#
# 【缺点】
#   - 每次摘要至少 **多一次完整 LLM 调用**；长历史时摘要输入本身也可能很贵。
#   - 摘要质量依赖 prompt；摘要错误会污染后续轮次，需要监控与回滚策略。
#   - 与 OpenAI compaction **机制不同**，不能假设二者信息保留能力一致。
#
# 【适用】多供应商、要强审计/可读小结、或 OpenAI 不可用时的主压缩手段。
# =============================================================================


class ChatRunnable(Protocol):
    """协议占位：任何具备 invoke(messages)->AIMessage 的对象均可（如 ChatAnthropic）。"""

    def invoke(self, messages: list[Any]) -> Any: ...


def llm_summarize_history_generic(
    *,
    full_history: list[Any],
    summarizer: ChatRunnable,
    keep_last_n: int = 12,
    system_prompt_for_summary: str = (
        "你是对话压缩助手。将下列较早的对话压缩为一条简短中文摘要，"
        "保留：用户目标、已确认事实、未决问题、关键工具结果要点。不要编造。"
    ),
) -> list[Any]:
    """
    将 full_history 中「除最后 keep_last_n 条外」的早期部分交给 summarizer 压成一条摘要消息，
    再拼接保留的尾部消息。返回新的消息列表（不修改入参）。

    full_history: 建议使用 LangChain HumanMessage / AIMessage / ToolMessage。
    summarizer: 任意供应商的 chat 模型。
    """
    # 延迟导入，避免无 LangChain 环境时报错
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if len(full_history) <= keep_last_n:
        return list(full_history)

    head = full_history[:-keep_last_n]
    tail = full_history[-keep_last_n:]

    # 将 head 序列化为模型可读的简单文本（生产可换结构化 XML）
    transcript = "\n".join(
        f"{getattr(m, 'type', m.__class__.__name__)}: {getattr(m, 'content', m)}"
        for m in head
    )

    summary_messages = [
        SystemMessage(content=system_prompt_for_summary),
        HumanMessage(content=transcript),
    ]
    summary_ai: AIMessage = summarizer.invoke(summary_messages)
    # 用一条「系统/人类可读」的摘要占位消息承接早期上下文（具体 role 按你栈约定）
    compressed_head: list[Any] = [
        HumanMessage(
            content=f"[Earlier conversation summary]\n{summary_ai.content}"
        )
    ]
    return compressed_head + list(tail)


# =============================================================================
# 方案六：LangChain Agent — SummarizationMiddleware（摘要模型可与主模型不同厂）
# =============================================================================
#
# 【优点】
#   - 与 create_agent 集成；接近窗口上限时自动摘要，减少手写调度代码。
#   - trigger/keep/trim_tokens_to_summarize 可调；摘要模型可换非 OpenAI。
#
# 【缺点】
#   - 额外 LLM 调用与中间件复杂度；升级 LangChain 需回归。
#   - 不解决「单条 tool 消息极大」的根本问题（需工具层配合）。
#   - 若与 OpenAI compaction 同时开，易重复压缩 → 费用与调试成本上升。
#
# 【适用】Agent 多轮 + 已用 LangChain 1.x agent 中间件栈。
# =============================================================================


def summarization_middleware_pattern() -> None:
    """概念示例：具体 import 以你安装的 langchain 版本为准。"""
    # from langchain.agents import create_agent
    # from langchain.agents.middleware import SummarizationMiddleware
    # from langchain_openai import ChatOpenAI
    # from langchain_anthropic import ChatAnthropic  # 示例：摘要用 Claude
    #
    # main_model = ChatOpenAI(model="gpt-4.1")
    # summary_model = ChatAnthropic(model="claude-3-5-haiku-latest")  # 非 OpenAI 摘要
    #
    # agent = create_agent(
    #     model=main_model,
    #     tools=[],
    #     middleware=[
    #         SummarizationMiddleware(
    #             model=summary_model,
    #             trigger=[("fraction", 0.85), ("messages", 60)],
    #             keep=("messages", 20),
    #             trim_tokens_to_summarize=8000,
    #         ),
    #     ],
    # )
    # _ = agent  # noqa: F841
    pass


# =============================================================================
# 方案对比小结（复制到设计文档用）
# =============================================================================
#
# 1) 目标仅为「少 token + 主要用 OpenAI」
#    → 优先评估：方案一或方案三（compaction + 链式 id）。
#
# 2) 目标为「payload 自控 + OpenAI」
#    → 方案二（items + 剪枝），承担更多工程与测试成本。
#
# 3) 目标为「不绑定 OpenAI 或要多厂」
#    → 方案四（窗口）作底线；方案五（通用摘要）作主力；方案六若在 Agent 内则用中间件。
#
# 4) 永远不要默认「开了 compaction 客户端列表就不会涨」
#    → 见主文档模式 A；要么链式 id，要么自研剪枝，要么改用手动摘要/窗口。
# =============================================================================


if __name__ == "__main__":
    # 本地快速验证「无 SDK」逻辑：滑动窗口 + 摘要列表拼接形状
    demo_msgs = [f"m{i}" for i in range(20)]
    assert len(sliding_window_messages(demo_msgs, max_messages=5)) == 5
    print("sliding_window_messages OK")
