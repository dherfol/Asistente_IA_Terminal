import warnings
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from typing import Optional
from tools import web_search, wikipedia_search

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

load_dotenv()  # Load environment variables from .env file

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: Optional[list[str]]



llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = """
You are an AI research assistant. Use the tools as possible.
Provide concise and accurate information on the given topic in spanish.
Return the result using the required structured format like the following:
- topic
- summary
- sources (string list of URLs)
- tools_used (if not: None)(If wikipedia_search was used, include 'Wikipedia'; if web_search was used, include 'DuckDuckGo', etc.)

Example format:
topic: ...
summary:
sources:
 - ...
tools_used:
 - ...
"""
tools = [web_search, wikipedia_search]

agent = create_agent(
    model = llm,
    tools=tools,
    system_prompt = prompt,
    response_format = ResearchResponse
)

question = input("¿Qué quieres que busque?: ")

raw_res = agent.invoke({
                "messages": [
                            {"role": "user", 
                            "content": question}
                ]
})

res = raw_res["structured_response"]

try:
    print(f"\nTema: {res.topic}\n")
    print(f"Resumen:\n{res.summary}\n")
    print("Fuentes:")
    for src in res.sources:
        print(f" - {src}")
    print("Herramientas usadas:")
    for t in res.tools_used:
        print(f" - {t}")
except Exception as e:
    print("Error al parsear la respuesta estructurada:", e)