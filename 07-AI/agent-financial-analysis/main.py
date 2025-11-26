# import necessary python libraries
import os
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.os import AgentOS

# create the AI finance agent
agent = Agent(
    name="xAI Finance Agent",
    model=OpenRouter(id="z-ai/glm-4.5-air:free",
                 api_key=os.getenv("OPENROUTER_API_KEY")),
    tools=[DuckDuckGoTools(), 
           YFinanceTools()],
    instructions = ["Always use tables to display financial/numerical data. For text data use bullet points and small paragrpahs."],
    debug_mode = True,
    markdown = True,
    )

agent.print_response("""How has the NVIDIA stock been performing since August 2025""", stream=True)


# # UI for finance agent
# agent_os = AgentOS(agents=[agent])
# app = agent_os.get_app()

# if __name__ == "__main__":
#     agent_os.serve(app="xai_finance_agent:app", reload=True)