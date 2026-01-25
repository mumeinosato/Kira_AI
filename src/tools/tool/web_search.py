import requests
from bs4 import BeautifulSoup
from ollama import Client
from config import OLLAMA_API_KEY
from src.tools.base_tool import BaseTool


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information"

    async def execute(self, query: str, memory_manager=None, **kwargs) -> str:
        result = await async_GoogleSearch(query)
        if result and "検索結果が見つかりませんでした" not in result and memory_manager:
            # Store search result as knowledge
            memory_manager.add_knowledge(result, source=f"web_search:{query}")
        return result

def custom_web_fetch(url, max_length=10000):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string if soup.title else ""

        for script in soup(["script", "style"]):
            script.decompose()
        raw_text = soup.get_text()
        content = ''.join(raw_text.split())

        title_clean = ''.join(title.split())
        if content.startswith(title_clean):
            content = content[len(title_clean):]

        if len(content) > max_length:
            content = content[:max_length]

        return {
            'title': title,
            'content': content,
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def async_GoogleSearch(query: str) -> str:
    client = Client(
        host='https://api.ollama.ai',
        headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
    )


    search_result = client.web_search(query)

    if search_result.results:
        first_url = search_result.results[0].url
        full_content = custom_web_fetch(first_url, max_length=4000)

        if full_content:
            return full_content['content']

    return "検索結果が見つかりませんでした"
