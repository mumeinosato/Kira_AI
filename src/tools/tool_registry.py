from typing import Dict, Type

from src.tools.base_tool import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self,tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, BaseTool | str]:
        return self._tools