import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

LISTENER_MODELS = os.environ.get(
    "LISTENER_MODELS",
    "mistralai/Ministral-3-14B-Reasoning,microsoft/Phi-4-reasoning-plus,Qwen/Qwen3-14B",
).split(",")

LISTENER_TEMPS = [
    float(t)
    for t in os.environ.get("LISTENER_TEMPS", "0.9,0.9,1.1").split(",")
]

TRUNCATION_PCTS = [
    float(p)
    for p in os.environ.get("TRUNCATION_PCTS", "0.25,0.50,0.75").split(",")
]

REWARD_TYPE = os.environ.get("REWARD_TYPE", "match_only")
CORRECTNESS_SCALAR = float(
    os.environ.get("CORRECTNESS_SCALAR", str(len(LISTENER_MODELS)))
)

_pool = None


def _init_listener_pool():
    global _pool
    if _pool is not None:
        return _pool

    from vllm import LLM, SamplingParams

    tp = int(os.environ.get("LISTENER_TP", "4"))
    gpu_util = float(os.environ.get("LISTENER_GPU_UTIL", "0.80"))

    _pool = []
    for model_name, temp in zip(LISTENER_MODELS, LISTENER_TEMPS):
        logger.info("Loading listener %s  temp=%.2f  tp=%d", model_name, temp, tp)
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
        )
        params = SamplingParams(
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=4096,
        )
        _pool.append((llm, params))
    return _pool

def _extract_thinking_and_answer(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = m.group(1).strip() if m else ""
    answer = ""
    idx = text.find("</think>")
    if idx != -1:
        answer = text[idx + len("</think>"):].strip()
    elif not reasoning:
        answer = text.strip()
    return reasoning, answer


def _truncate_reasoning(reasoning: str, pct: float) -> str:
    lines = [ln for ln in reasoning.split("\n") if ln.strip()]
    if not lines:
        return ""
    n = max(1, int(len(lines) * pct))
    return "\n".join(lines[:n])


_OPTION_RE = re.compile(
    r"(?:option|answer|choice)\s*[:is]*\s*([A-F])", re.IGNORECASE
)
_LETTER_RE = re.compile(r"\b([A-F])\b")
_KEYWORD_MAP = {"true": "A", "false": "C", "uncertain": "B"}


def _extract_answer_option(text):
    """Pull the first option letter (A-F) from answer text."""
    m = _OPTION_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    low = text.lower()
    for kw, opt in _KEYWORD_MAP.items():
        if kw in low:
            return opt
    return None

def compute_cross_execution_reward(
    speaker_answer: str,
    prompt_text: str,
    reasoning: str,
):
    pool = _init_listener_pool()
    total_matches = 0

    for pct in TRUNCATION_PCTS:
        truncated = _truncate_reasoning(reasoning, pct)
        if not truncated:
            continue
        listener_prompt = f"{prompt_text}\n<think>\n{truncated}\n"

        for llm, params in pool:
            outputs = llm.generate([listener_prompt], params)
            listener_text = outputs[0].outputs[0].text
            _, listener_answer_text = _extract_thinking_and_answer(listener_text)
            listener_opt = _extract_answer_option(listener_answer_text)
            if listener_opt and listener_opt == speaker_answer:
                total_matches += 1

    return float(total_matches)

def compute_balanced_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    reasoning, answer_text = _extract_thinking_and_answer(solution_str)
    speaker_answer = _extract_answer_option(answer_text)

    if not speaker_answer:
        return 0.0

    prompt_text = ""
    if extra_info:
        prompt_text = (extra_info if isinstance(extra_info, dict) else {}).get(
            "prompt_text", ""
        )

    if REWARD_TYPE == "match_only":
        if not reasoning:
            return 0.0
        return compute_cross_execution_reward(speaker_answer, prompt_text, reasoning)

    elif REWARD_TYPE == "balanced":
        cross_execution_reward = (
            compute_cross_execution_reward(speaker_answer, prompt_text, reasoning)
            if reasoning
            else 0.0
        )
        correctness_reward = float(speaker_answer == ground_truth)
        return cross_execution_reward + CORRECTNESS_SCALAR * correctness_reward

    elif REWARD_TYPE == "correctness_only":
        return float(speaker_answer == ground_truth)

    else:
        raise ValueError(f"Unknown REWARD_TYPE={REWARD_TYPE}")

def compute_score_correctness_only(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    _, answer_text = _extract_thinking_and_answer(solution_str)
    opt = _extract_answer_option(answer_text)
    return float(opt == ground_truth) if opt else 0.0
