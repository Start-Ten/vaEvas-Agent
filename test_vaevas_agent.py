"""Test vaevas_agent with DeepSeek anthropic-compatible."""
import os, sys
sys.path.insert(0, ".")

# Read API key from environment or Claude Code settings
_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
if not _api_key:
    print("Set ANTHROPIC_API_KEY env var to run this test")
    sys.exit(1)
os.environ["ANTHROPIC_API_KEY"] = _api_key

from vaevas_agent.config import LLMConfig
from vaevas_agent.llm.client import call_llm

config = LLMConfig(
    provider="anthropic-compatible",
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/anthropic",
    max_tokens=4096,          # enough for thinking + code
    temperature=0.0,
)

system = "You are a Verilog-A expert. Reply ONLY with code blocks. NO explanation."
user = "Write a minimal Verilog-A NOT gate module: ports VDD/VSS/A/Y. Only output the code."

print("[test] vaevas_agent -> DeepSeek deepseek-v4-flash")
resp = call_llm(config, system, user)
print(f"OK  {resp.elapsed_ms:.0f}ms  in={resp.input_tokens} out={resp.output_tokens}")
print(f"text={len(resp.text)} chars")
print("---")
# Write to file to avoid terminal encoding issues
with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(resp.text)
print("Response saved to test_output.txt")
print("DONE")