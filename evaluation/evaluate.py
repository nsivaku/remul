import argparse
import logging
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.evaluation_dataset import Dataset
from utils.trace_utils import extract_thinking_and_answer
from metrics import evaluate_accuracy, evaluate_hint_usage, evaluate_truncated_cot_aoc, evaluate_adding_mistakes_aoc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(
        self,
        model_path,
        temperature = 0.7,
        max_new_tokens = 8192,
        device = 0
    ):
        logger.info("Loading model from: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def __call__(
        self,
        prompt: str,
        prefix: str = None,
    ):
        if prefix is not None:
            full_prompt = f"{prompt}\n<think>\n{prefix}\n"
        else:
            full_prompt = prompt

        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        full_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        reasoning, answer = extract_thinking_and_answer(full_text)
        if prefix and reasoning:
            reasoning = prefix + "\n" + reasoning
        return reasoning, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", choices=["bbeh", "zlb", "musr", "folio"], required=True)
    parser.add_argument("--metric", choices=["accuracy", "hint_usage", "truncated_cot_aoc", "adding_mistakes_aoc"], required=True)
    args = parser.parse_args()

    if args.metric == "accuracy":
        metric = evaluate_accuracy
    elif args.metric == "hint_usage":
        metric = evaluate_hint_usage
    elif args.metric == "truncated_cot_aoc":
        metric = evaluate_truncated_cot_aoc
    elif args.metric == "adding_mistakes_aoc":
        metric = evaluate_adding_mistakes_aoc

    model_evaluator = ModelEvaluator(args.model_path, args.temperature, args.max_new_tokens, args.device)
    dataset = Dataset(args.dataset)
    results = metric(model_evaluator, dataset.data)
    with open(f"results/{args.dataset}_{args.metric}_{args.model_path}.json", "w") as f:
        json.dump(results, f)