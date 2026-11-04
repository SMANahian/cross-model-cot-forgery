"""Pre-download every model's weights to HF_HOME (container disk) on the pod.

The first three matrix models come first (needed for stage 1), then the rewriter, then
the two models added in v6. Duplicate weight formats (gpt-oss `original/` and `metal/`,
Mistral's `consolidated.safetensors`) are skipped here, though vLLM fetches Mistral's
copy anyway when it loads the rewriter.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

FIRST = {  # downloaded in parallel
    "openai/gpt-oss-20b": ["original/*", "metal/*"],
    "Qwen/Qwen3-8B": [],
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": [],
}
THEN = {  # downloaded one after another
    "mistralai/Mistral-Small-24B-Instruct-2501": ["consolidated*"],
    "allenai/Olmo-3-7B-Think": [],
    "zai-org/GLM-Z1-9B-0414": [],
}


def fetch(repo, ignore):
    t0 = time.time()
    try:
        path = snapshot_download(repo, ignore_patterns=ignore or None)
        print(f"OK   {repo} in {time.time() - t0:.0f}s -> {path}", flush=True)
    except Exception as e:  # e.g. a gated repo; report and keep going
        print(f"FAIL {repo}: {type(e).__name__}: {e}", flush=True)


with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(lambda kv: fetch(*kv), FIRST.items()))
for repo, ignore in THEN.items():
    fetch(repo, ignore)
print("ALL DOWNLOADS FINISHED", flush=True)
