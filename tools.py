from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from datetime import datetime

@tool
def wikipedia_search(query: str) -> str:
    """Busca resultados en Wikipedia."""
    return WikipediaAPIWrapper().run(query)

@tool
def web_search(query: str) -> str:
    """Busca en la web usando DuckDuckGo."""
    return DuckDuckGoSearchRun().run(query)