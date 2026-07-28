from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import phi
import phi.api
import os
from phi.playground import Playground,serve_playground_app

## Load environment Variables from .env file
from dotenv import load_dotenv
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

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

app=Playground(agents=[finance_agent,websearch_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)
