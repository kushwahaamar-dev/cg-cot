
from src.benchmarks import load_math
from src.llm_client import make_client
from src.methods.cot import COT_SYSTEM
import time

def debug_math():
    print("Loading MATH...")
    problems = load_math(n=1)
    p = problems[0]
    print(f"Problem 0: {p.id}")
    print(f"Text length: {len(p.text)}")
    print(f"Text preview: {p.text[:200]}...")
    
    client = make_client("ollama", model="qwen2.5:7b")
    print("Calling LLM...")
    start = time.time()
    try:
        # call_raw is what CoT uses
        resp = client.call_raw(COT_SYSTEM, p.text)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")
        print(f"Output length: {len(resp)}")
        print(f"Output preview: {resp[:200]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_math()
