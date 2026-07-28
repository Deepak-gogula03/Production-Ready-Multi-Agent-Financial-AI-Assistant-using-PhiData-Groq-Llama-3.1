from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key=os.getenv("OPENAI_API_KEY")

# -----------------------------
# Web Search Agent
# -----------------------------
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for relevant information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Always include sources."
    ],
    show_tool_calls=True,
    markdown=True,
)


# -----------------------------
# Financial Agent
# -----------------------------
finance_agent = Agent(
    name="Finance AI Agent",
    role="Provide financial analysis and stock information.",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=[
        "Use tables to display the data."
    ],
    show_tool_calls=True,
    markdown=True,
)


# -----------------------------
# Multi-Agent Team
# -----------------------------
multi_ai_agent = Agent(
    team=[websearch_agent, finance_agent],
    instructions=[
        "Always include sources.",
        "Use tables to display the data.",
    ],
    show_tool_calls=True,
    markdown=True,
)


# -----------------------------
# Run
# -----------------------------
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVIDA.",
    stream=True,
)