from __future__ import annotations

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    return f"""You are an electronics order assistant. Today is {today or '2026-06-01'}.
You ONLY manage electronics orders. Refuse any requests for travel planning, fake invoices, manual discount overrides, or bypassing stock checks. Do not ignore catalog or policy.

Before processing an order, ensure you have ALL of these:
- customer name
- phone number
- email
- shipping address
- at least one product request with quantity
If any are missing, clarify and stop immediately.

When you have all information, you MUST use the following tools SEQUENTIALLY, one by one. NEVER call them in parallel. After calling a tool, wait for its result before calling the next tool:
1. Call `list_products` to find items. Wait for result.
2. Call `get_product_details` with the discovered product IDs. You will receive a `detail_token`. Wait for result.
3. Call `get_discount` using the customer email as `seed_hint`. You will receive a `discount_rate` and `campaign_code`. Wait for result.
4. Call `calculate_order_totals` using the exact `items`, `detail_token`, and `discount_rate`. Wait for result.
5. Call `save_order` using all the above information.

NEVER hallucinate `detail_token`, `product_ids`, or `discount_rate`. You MUST get them from the tool outputs.

Only use tool outputs for product IDs, prices, stock, discount, totals, and save path. Do not hallucinate or make up details.
Return one concise final answer in Vietnamese acknowledging the order or explaining the failure."""


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(**kwargs) -> str:
        """Search the local product catalog and return the best matching items."""
        return json.dumps(store.list_products(**kwargs), ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(**kwargs) -> str:
        """Return the simulated campaign discount for the order."""
        return json.dumps(store.get_discount(**kwargs), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(**kwargs) -> str:
        """Validate stock and calculate the discounted order total."""
        return json.dumps(store.calculate_order_totals(**kwargs), ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(**kwargs) -> str:
        """Persist the final order to a local JSON file."""
        return json.dumps(store.save_order(**kwargs), ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    
    # Disable parallel tool calling globally for Groq to fix Llama 8B
    if provider == "groq":
        from langchain_groq import ChatGroq
        if not hasattr(ChatGroq, "_original_bind_tools"):
            ChatGroq._original_bind_tools = ChatGroq.bind_tools
            def patched_bind_tools(self, tools, **kwargs):
                if "parallel_tool_calls" not in kwargs:
                    kwargs["parallel_tool_calls"] = False
                return self._original_bind_tools(tools, **kwargs)
            ChatGroq.bind_tools = patched_bind_tools
            
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending = {}
    records = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
