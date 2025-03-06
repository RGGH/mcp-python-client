import asyncio
import json
import os
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

MODEL = config["model"]
API_KEY = config["api_key"]
BASE_URL = config["base_url"]
SYSTEM_PROMPT = config["system_prompt"]

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        print(f"Connecting to MCP server: {self.server_params.command} {' '.join(self.server_params.args) if self.server_params.args else ''}")
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()
        print(f"Successfully connected to MCP server")

    async def get_available_tools(self):
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        return await self.session.list_tools()

    def call_tool(self, tool_name: str) -> Any:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        async def callable(*args, **kwargs):
            print(f"Calling tool: {tool_name} with args: {kwargs}")
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable

async def agent_loop(query: str, tools: dict, messages: List[dict] = None):
    messages = messages or [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            tools="\n- ".join(
                [f"{t['name']}: {t['schema']['function']['description']}" for t in tools.values()]
            )
        )}
    ]
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[t["schema"] for t in tools.values()] if tools else None,
        max_tokens=4096,
        temperature=0,
    )

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_result = await tools[tool_call.function.name]["callable"](**arguments)
            messages.extend([
                response.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": json.dumps(tool_result)},
            ])
        response = await client.chat.completions.create(model=MODEL, messages=messages)

    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content, messages

async def main():
    try:
        print("Starting MCP client...")
        print(f"Config: {json.dumps(config, indent=2)}")
        
        mcp_clients = []
        tools = {}

        for server in config["mcp_servers"]:
            print(f"Setting up server: {server['name']}")
            server_params = StdioServerParameters(
                command=server["command"],
                args=server["args"],
                env=server["env"],
            )
            client = MCPClient(server_params)
            mcp_clients.append(client)
            async with client as mcp_client:
                tools_data = await mcp_client.get_available_tools()
                print(f"Retrieved tools from {server['name']}: {tools_data}")
                
                if isinstance(tools_data, tuple) and len(tools_data) >= 2:
                    for tool in tools_data[1]:
                        tools[tool.name] = {
                            "name": tool.name,
                            "callable": mcp_client.call_tool(tool.name),
                            "schema": {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.inputSchema,
                                },
                            },
                        }
        
        print(f"Available tools: {', '.join(tools.keys())}")
        
        messages = None
        while True:
            try:
                user_input = input("\nEnter your prompt (or 'quit' to exit): ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                response, messages = await agent_loop(user_input, tools, messages)
                print("\nResponse:", response)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError in prompt loop: {e}")
    except Exception as e:
        print(f"\nError in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
