"""Shared pieces: model registry, prompt rendering, output parsing, JSONL helpers.

Every model is rendered with its OWN chat template (tokenizer.apply_chat_template),
so the role tags are exactly the ones the model was trained on. The whole benchmark
is about roles, so we never hand-write template tokens.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TOKENS = 8192        # room for long reasoning; outputs that hit it are "incomplete"
MAX_MODEL_LEN = 12288    # prompt (<1k tokens) + MAX_TOKENS

# Decoding follows each model card's recommended sampling (proposal 3.6):
# Qwen3 and the R1 distills loop endlessly under greedy decoding in thinking mode.
MODELS = {
    "gpt-oss-20b": {
        "hf_id": "openai/gpt-oss-20b",
        "format": "harmony",  # reasoning in the `analysis` channel, answer in `final`
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "template_kwargs": {"reasoning_effort": "high"},  # the paper used max effort
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "format": "think",  # <think> reasoning </think> answer
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "template_kwargs": {"enable_thinking": True},
    },
    "r1-llama-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "format": "think",
        "sampling": {"temperature": 0.6, "top_p": 0.95},
        "template_kwargs": {},
        # Its template puts system text untagged before the user turn, and with the rule
        # there it leaked 9/10 with no forgery (it read the code as user-provided). Its
        # card says to put all instructions in the user prompt, so (pre-registered
        # contingency, approved): rule in a first user turn + assistant acknowledgement.
        "rule_placement": "first_user_turn",
    },
    # Stretch goal (proposal 3.2). Flag: its card prescribes its own reasoning system
    # prompt, so attack prompts need a decision before this model is used as a target.
    "phi4-reasoning": {
        "hf_id": "microsoft/Phi-4-reasoning",
        "format": "think",
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
        "template_kwargs": {},
    },
}

# Rewriter (proposal 3.2): outside every tested family, so its own style can't favor a column.
REWRITER = {"hf_id": "mistralai/Mistral-Small-24B-Instruct-2501"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(cfg):
    """The tokenizer we render prompts with and decode outputs with.

    Not llm.get_tokenizer(): for the R1 distill, vLLM 0.33 + transformers 5 builds a
    legacy LlamaTokenizer from tokenizer_config's class name, which ENCODES Llama-3
    text differently and decodes to byte-level pieces ("Ġ", "Ċ"). AutoTokenizer builds
    the correct tokenizer from tokenizer.json (for Qwen3 and gpt-oss both agree).
    vLLM then only ever sees token ids.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    probe = "Okay, so the user asks.\nWe need to check it."
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == probe, f"tokenizer round-trip failed for {cfg['hf_id']}"
    return tok


def decode_output(tokenizer, completion):
    """Decode a vLLM completion from its token ids with our HF tokenizer
    (never vLLM's own detokenized text, see load_tokenizer)."""
    text = tokenizer.decode(list(completion.token_ids), skip_special_tokens=False)
    if text.count("Ġ") > 5 and text.count(" ") < text.count("Ġ"):
        raise RuntimeError("decoded text still contains byte-level BPE pieces")
    return text


def render(tokenizer, messages, cfg):
    """Render `messages` with the model's own chat template.

    Returns (prompt_text, prompt_token_ids). We tokenize the rendered text with
    add_special_tokens=False because the template already contains the BOS and
    role tokens; letting the tokenizer add another BOS would corrupt the prompt.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **cfg["template_kwargs"]
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, ids


# Special tokens like <|im_end|>, <|return|> or DeepSeek's full-width <｜end▁of▁sentence｜>.
_SPECIAL = re.compile(r"<\|[^|>]{1,40}\|>|<｜[^｜>]{1,40}｜>")


def _clean(s):
    return None if s is None else _SPECIAL.sub("", s).strip()


def parse_output(raw, fmt):
    """Split a raw completion (generated with skip_special_tokens=False) into
    (reasoning, final_answer). final_answer is None when the model never reached
    its answer, e.g. it ran out of tokens mid-reasoning.
    """
    if fmt == "harmony":
        # gpt-oss: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
        analysis = re.findall(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)", raw, re.S
        )
        final = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|$)", raw, re.S
        )
        reasoning = "\n".join(a.strip() for a in analysis)
        answer = final.group(1) if final else None
    else:
        # <think> models. The R1 template may already put "<think>\n" in the prompt,
        # so the opening tag is optional; the closing tag marks the answer.
        if "</think>" in raw:
            reasoning, answer = raw.split("</think>", 1)
        else:
            reasoning, answer = raw, None
        reasoning = reasoning.replace("<think>", "", 1)
    return _clean(reasoning), _clean(answer)
