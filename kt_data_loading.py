import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from prompting import kt_system_prompt, kt_user_prompt_unpacked, kt_user_prompt_packed
from utils import device

def apply_annotations(sample: dict):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    # Handle dialogues beginning with turn 0 (student-initiated)
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    # Copy correctness and kcs into dialogue
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        dia_turn["correct"] = anno_turn["correct"]
        dia_turn["kcs"] = anno_turn["kcs"]
    # Use human annotation of correctness for final turn
    if "expected_result" in sample["meta_data"]: # CoMTA
        dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
    elif "self_correctness" in sample["meta_data"]: # MathDial
        if dialogue[-1]["correct"] is not None: # Final turn could be closing remarks, so skip if not tagged as having correctness
            if sample["meta_data"]["self_correctness"] == "Yes":
                dialogue[-1]["correct"] = True
            elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                dialogue[-1]["correct"] = None
            elif sample["meta_data"]["self_correctness"] == "No":
                dialogue[-1]["correct"] = False
    return dialogue

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class KTDatasetUnpacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            for turn_idx, turn in enumerate(dialogue):
                if turn["correct"] is None:
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt_unpacked(sample, turn_idx, kc, args)},
                            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                        ], tokenize=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")

class KTCollatorUnpacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
            "num_kcs": [len(sample["prompts"]) for sample in batch],
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }

class KTDatasetPacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            for turn_idx, turn in enumerate(dialogue):
                if turn["correct"] is None:
                    continue
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": kt_system_prompt(args)},
                    {"role": "user", "content": kt_user_prompt_packed(sample, turn_idx, args)},
                ], tokenize=False)
                kc_conts = [
                    tokenizer.apply_chat_template([
                        {"role": "user", "content": kc},
                        {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                    ], tokenize=False)
                    for kc in turn["kcs"]
                ]
                kc_conts = [" " + cont.split("user<|end_header_id|>\n\n")[1] for cont in kc_conts]
                prompt = prompt + "".join(kc_conts)
                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")

class KTCollatorPacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = prompts_tokenized.input_ids.to(device)
        batch_size, max_seq_len = input_ids.shape
        eos_idxs = [
            (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero().squeeze().cpu()
            for seq_idx in range(batch_size)
        ]
        # Create default lower triangular 4D attention mask
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        # Create default 2D position id matrix
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        # Set attention mask and position ids for each sequence
        for seq_idx in range(batch_size):
            # Get end of context
            context_end_idx = eos_idxs[seq_idx][1]
            # Initialize to no attention to any tokens after context
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            # Update attention mask and position ids for each KC
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                # Set position ids as if KC immediately followed context
                position_ids[seq_idx, start_idx : end_idx + 1] = torch.arange(context_end_idx, context_end_idx + end_idx - start_idx + 1)
                # Set KC attention mask to lower triangular to permit self-attention
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                # Go to next KC
                start_idx = end_idx + 1

        # Get index of token before eos for each KC, pad for easier loss computation
        last_idxs = pad_sequence([idxs[3::2] - 1 for idxs in eos_idxs], batch_first=True)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.unsqueeze(1), # Add singleton head dimension
            "position_ids": position_ids,
            "last_idxs": last_idxs,
            "num_kcs": [len(sample["kcs"]) for sample in batch],
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }

def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
