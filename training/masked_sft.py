import os
import re
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MaskedSFTDataset(Dataset):


    def __init__(
        self,
        prompts: List[str],
        ground_truths: List[str],
        responses: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for prompt, gt, response in zip(prompts, ground_truths, responses):
            item = self._process(prompt, gt, response)
            if item is not None:
                self.items.append(item)

        logger.info(f"Created MaskedSFTDataset with {len(self.items)} examples")

    def _process(self, prompt: str, gt: str, response: str) -> Optional[Dict]:
        reasoning, _ = _extract_thinking_and_answer(response)
        corrected = f"<think>\n{reasoning}\n</think>\n{gt}"
        full_text = prompt + corrected

        encoding = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze(0)

        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]

        gen_text = self.tokenizer.decode(input_ids[prompt_len:])
        think_end = gen_text.find("</think>")

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if think_end != -1:
            pre_answer = gen_text[:think_end + len("</think>")]
            pre_tokens = self.tokenizer(pre_answer, return_tensors="pt")
            answer_start = prompt_len + pre_tokens.input_ids.shape[1]
            answer_start = min(answer_start, len(input_ids) - 1)
            loss_mask[answer_start:] = 1.0
        else:
            gen_len = len(input_ids) - prompt_len
            answer_start = prompt_len + int(gen_len * 0.8)
            loss_mask[answer_start:] = 1.0

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "labels": input_ids.clone(),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, masks, labels, attn = [], [], [], []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        masks.append(F.pad(item["loss_mask"], (0, pad_len), value=0.0))
        labels.append(F.pad(item["labels"], (0, pad_len), value=-100))
        attn.append(torch.cat([torch.ones(seq_len), torch.zeros(pad_len)]))

    return {
        "input_ids": torch.stack(input_ids),
        "loss_mask": torch.stack(masks),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }

def _extract_thinking_and_answer(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = m.group(1).strip() if m else ""
    answer = ""
    end = text.find("</think>")
    if end != -1:
        answer = text[end + len("</think>"):].strip()
    return reasoning, answer


@torch.no_grad()
def generate_speaker_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 8192,
    batch_size: int = 4,
) -> List[str]:
    """Generate CoT responses from the faithfulness-trained speaker model."""
    model.eval()
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=4096,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, output in enumerate(outputs):
            gen = output[inputs.input_ids.shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            responses.append(text)

        if (i // batch_size) % 10 == 0:
            logger.info(f"Generated {min(i + batch_size, len(prompts))}/{len(prompts)} responses")

    return responses

def train_masked_sft(
    model_path: str,
    train_parquet: str,
    output_dir: str,
    lora_rank: int = 32,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-5,
    num_epochs: int = 5,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading faithfulness-trained model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading training data from: {train_parquet}")
    df = pd.read_parquet(train_parquet)
    prompts_raw = df["prompt"].tolist()
    ground_truths = df["ground_truth"].tolist()

    prompts = []
    for p in prompts_raw:
        msgs = json.loads(p) if isinstance(p, str) else p
        text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        prompts.append(text)

    logger.info("Generating speaker responses...")
    responses = generate_speaker_responses(model, tokenizer, prompts)

    logger.info("Applying LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = MaskedSFTDataset(prompts, ground_truths, responses, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = num_epochs * len(dataloader) // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Training loop ----
    logger.info(f"Starting masked SFT training for {num_epochs} epochs")
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = loss_mask[:, 1:].contiguous()

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            masked_loss = per_token_loss * shift_mask
            n_answer = shift_mask.sum()
            loss = masked_loss.sum() / n_answer.clamp(min=1)
            loss = loss / gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg = epoch_loss / (step + 1)
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                        f"Loss: {avg:.4f} | Answer tokens: {n_answer.item():.0f}"
                    )

    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")

    merged_model = model.merge_and_unload()
    merged_path = os.path.join(output_dir, "merged")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Saved merged REMUL model to {merged_path}")

    return merged_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to baseline checkpoint")
    parser.add_argument("--train_parquet", type=str, required=True,
                        help="Path to training data parquet")
    parser.add_argument("--output_dir", type=str, default="./outputs/masked_sft")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    train_masked_sft(
        model_path=args.model_path,
        train_parquet=args.train_parquet,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
