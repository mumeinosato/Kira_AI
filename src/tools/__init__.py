from src.tools.tool.web_search import WebSearchTool
from src.tools.tool_registry import ToolRegistry


def register_default_tools(registry: ToolRegistry):
    registry.register(WebSearchTool())