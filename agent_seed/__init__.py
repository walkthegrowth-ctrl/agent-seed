"""agent_seed: ultra minimalistic LLM agent with tool calling.

Layout:
    agent_seed.core       agent loop (history, LLM call, tool dispatch)
    agent_seed.llm        thin streaming client for OpenAI-compatible APIs
    agent_seed.tools      tool registry + JSON schema generation
    agent_seed.backends   HTTP/SSE backend (server)
    agent_seed.frontends  REPL terminal frontend (future: web, tui, ...)
"""

__version__ = "0.1.0"
