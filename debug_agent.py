import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import build_agent
from grade.scoring import load_cases

def debug():
    cases = load_cases(Path("data/graded_cases.json"))
    query = cases[0]["query"]
    
    agent = build_agent(
        provider="fireworks",
        model_name="accounts/fireworks/models/deepseek-v4-pro"
    )
    
    print("QUERY:", query)
    print("-" * 50)
    
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    for msg in response["messages"]:
        print(f"[{msg.type.upper()}]")
        if hasattr(msg, "content") and msg.content:
            print("Content:", msg.content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("Tool Calls:", msg.tool_calls)
        print("-" * 50)

if __name__ == "__main__":
    debug()
