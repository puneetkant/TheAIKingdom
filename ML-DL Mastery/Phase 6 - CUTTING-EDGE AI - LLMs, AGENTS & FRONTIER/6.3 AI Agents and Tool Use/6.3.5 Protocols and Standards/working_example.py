"""
Working Example: Protocols and Standards for AI Agents
Covers MCP (Model Context Protocol), A2A, agent interoperability,
and standardised communication patterns.
"""
import json, os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_protocols")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Model Context Protocol (MCP) ------------------------------------------
def mcp_overview():
    print("=== Model Context Protocol (MCP) ===")
    print()
    print("  Anthropic (2024) open standard for LLM <-> tool/data source communication")
    print("  Analogy: USB-C for AI — one connector for all data sources")
    print()
    print("  Three primitives:")
    primitives = [
        ("Resources",  "Files, databases, APIs exposed as data sources"),
        ("Tools",      "Functions the LLM can call (like function calling)"),
        ("Prompts",    "Reusable prompt templates and workflows"),
    ]
    for p, d in primitives:
        print(f"  {p:<12} {d}")
    print()
    print("  Transport layers:")
    transports = [
        ("stdio",       "Local: server is a subprocess; stdin/stdout JSON-RPC"),
        ("SSE over HTTP","Remote: server-sent events; web-accessible MCP server"),
    ]
    for t, d in transports:
        print(f"  {t:<14} {d}")

    print()
    print("  Example MCP server (Python, illustrative):")
    mcp_code = '''
from mcp import Server, Resource, Tool

server = Server("my-weather-server")

@server.resource("weather://{location}")
def get_weather_resource(location: str) -> str:
    """Current weather data for a location."""
    return fetch_weather_api(location)

@server.tool()
def search_historical_weather(location: str, date: str) -> dict:
    """Search historical weather records."""
    return query_weather_db(location, date)

server.run()  # starts stdio or HTTP transport
'''
    print(mcp_code)

    print("  MCP ecosystem (2025):")
    print("    Official servers: GitHub, Slack, Google Drive, Postgres, filesystem")
    print("    Clients: Claude Desktop, Cline, Cursor, Continue, LangChain")
    print("    Registry: mcp.so (community server discovery)")


# -- 2. Agent-to-Agent (A2A) protocol -----------------------------------------
def a2a_protocol():
    print("\n=== Agent-to-Agent (A2A) Protocol ===")
    print()
    print("  Google (2025) open standard for agent interoperability")
    print("  Enables different vendor agents to communicate")
    print()
    print("  Core concepts:")
    concepts = [
        ("Agent Card",    "JSON descriptor: capabilities, endpoint, auth"),
        ("Tasks",         "Unit of work sent from client to remote agent"),
        ("Messages",      "Turn-based message exchange within a task"),
        ("Parts",         "Typed content: text, file, data (structured)"),
        ("Artifacts",     "Output objects produced by the agent"),
        ("Events",        "SSE streaming of task progress"),
    ]
    for c, d in concepts:
        print(f"  {c:<14} {d}")
    print()
    print("  Sample Agent Card:")
    agent_card = {
        "name": "weather-agent",
        "description": "Provides weather information and forecasts",
        "version": "1.0.0",
        "url": "https://weather-agent.example.com/a2a",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
        "skills": [
            {"id": "current-weather", "name": "Current Weather",
             "description": "Get current conditions for any location"},
            {"id": "forecast", "name": "5-Day Forecast",
             "description": "Get 5-day weather forecast"},
        ]
    }
    print(json.dumps(agent_card, indent=2))


# -- 3. OpenAI Swarm handoff pattern ------------------------------------------
def swarm_handoff():
    print("\n=== Handoff Pattern (OpenAI Swarm) ===")
    print()
    print("  Agents hand off to specialised agents when needed")
    print("  Lightweight, stateless — suitable for learning/prototyping")
    print()
    swarm_code = '''
from swarm import Swarm, Agent

client = Swarm()

billing_agent = Agent(
    name="Billing Agent",
    instructions="Handle billing questions. Escalate refunds to manager.",
    functions=[get_invoice, process_payment],
)

support_agent = Agent(
    name="Support Agent",
    instructions="Help with technical issues. Hand billing to Billing Agent.",
    functions=[check_status, restart_service,
               lambda: billing_agent],   # handoff function
)

# Multi-agent conversation
response = client.run(
    agent=support_agent,
    messages=[{"role": "user", "content": "I need a refund for my last invoice"}],
)
print(response.agent.name)  # -> "Billing Agent" (after handoff)
'''
    print(swarm_code)

    print("  Handoff design guidelines:")
    guidelines = [
        "Define clear boundaries between agent responsibilities",
        "Include context in handoff (don't lose conversation history)",
        "Allow agents to reject handoffs (backtrack)",
        "Log all handoffs for debugging and evaluation",
        "Test handoff triggers explicitly",
    ]
    for g in guidelines:
        print(f"  • {g}")


# -- 4. Standardisation landscape ----------------------------------------------
def standards_landscape():
    print("\n=== Agent Standards and Interoperability Landscape ===")
    print()
    standards = [
        ("MCP",             "Anthropic; LLM <-> tools/data; widely adopted 2025"),
        ("A2A",             "Google; agent <-> agent; cross-vendor"),
        ("OpenAI plugins",  "Deprecated; replaced by function calling"),
        ("AgentConnect",    "Microsoft; agent registry in Azure"),
        ("OCSF",            "Security events; threat sharing across agents"),
        ("JSON-RPC 2.0",    "Base protocol for MCP messaging"),
        ("OpenAPI 3.1",     "Tool schemas; auto-generate MCP servers from APIs"),
    ]
    print(f"  {'Standard':<18} {'Notes'}")
    for s, d in standards:
        print(f"  {s:<18} {d}")
    print()
    print("  Key principle: agents should be composable building blocks")
    print("  -> Design agents with well-defined interfaces, not monolithic pipelines")


if __name__ == "__main__":
    mcp_overview()
    a2a_protocol()
    swarm_handoff()
    standards_landscape()
