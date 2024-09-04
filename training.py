import json
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from models import get_model, get_checkpoint_path
from data_loading import load_annotated_data, get_kc_result_filename
from kt_data_loading import KTDatasetUnpacked, KTCollatorUnpacked, KTDatasetPacked, KTCollatorPacked, get_dataloader
from prompting import get_true_false_tokens
from utils import device

PACK_KCS = True

def get_loss_unpacked(model, batch, true_token, false_token):
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
    # Get probability that all KCs are True for each turn in the batch
    num_kc_counter = 0
    kc_probs_grouped = []
    corr_probs = []
    for num_kcs in batch["num_kcs"]:
        kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
        corr_probs.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod())
        num_kc_counter += num_kcs
    corr_probs = torch.stack(corr_probs)
    # Get BCE loss with correctness labels and predicted probabilities
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs

def get_loss_packed(model, batch, true_token, false_token):
    # Invert attention mask
    attention_mask = batch["attention_mask"]
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model.dtype)
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=attention_mask, position_ids=batch["position_ids"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
    kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
    # Get probability that all KCs are True for each turn in the batch
    kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])]
    # Set probs to 1 on padded indices
    kc_probs = torch.masked_scatter(kc_probs, batch["last_idxs"].to(device) == 0, torch.ones_like(kc_probs).to(device))
    # Get BCE loss with correctness labels and predicted probabilities
    corr_probs = kc_probs.prod(dim=1)
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs

def train(args):
    assert args.model_name

    if args.crossval:
        model_name = args.model_name
        metrics_agg = []
        for fold in range(5):
            print(f"Running fold {fold + 1}...")
            args.model_name = f"{model_name}_{fold + 1}"
            metrics = train_fold(args, fold)
            metrics_agg.append(metrics)
        metrics_np = np.array(metrics_agg)
        avg = metrics_np.mean(axis=0)
        std = metrics_np.std(axis=0)
        results = [
            f"{metric}: {avg[idx]:.4f} \\pm {std[idx]:.4f}" for idx, metric in
            enumerate(["Loss", "Acc", "AUC", "Prec", "Rec", "F1", "Acc (Final)", "AUC (Final)", "Prec (Final)", "Rec (Final)", "F1 (Final)"])
        ]
        result_str = "\n".join(results)
        print(result_str)
        with open(f"results/metrics_crossval_{model_name}.txt", "w") as out_file:
            out_file.writelines([
                str(metrics_agg) + "\n",
                result_str + "\n"
            ])
    else:
        train_fold(args, 0)

def train_fold(args, fold: int):
    # Load language model with trainable LoRA adapters
    model, tokenizer = get_model(args.base_model, False, r=args.r, lora_alpha=args.lora_alpha)
    model.print_trainable_parameters()

    # Load and split dataset, annotated with correctness and KCs
    KTDataset = KTDatasetPacked if PACK_KCS else KTDatasetUnpacked
    KTCollator = KTCollatorPacked if PACK_KCS else KTCollatorUnpacked
    get_loss = get_loss_packed if PACK_KCS else get_loss_unpacked
    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    train_dataset = KTDataset(train_df, tokenizer, args)
    val_dataset = KTDataset(val_df, tokenizer, args)
    collator = KTCollator(tokenizer)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Do training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _, _ = get_loss(model, batch, true_token, false_token)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        # TODO: why is loss here different than when testing on val set?
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _, _ = get_loss(model, batch, true_token, false_token)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model.save_pretrained(get_checkpoint_path(args.model_name))
            best_val_loss = avg_val_loss

    return test(args, fold)

def compute_metrics(labels, preds):
    hard_preds = np.round(preds)
    acc = accuracy_score(labels, hard_preds)
    auc = roc_auc_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard_preds, average="binary")
    return acc, auc, prec, rec, f1

def test(args, fold: int = 0):
    # Load trained language model
    model, tokenizer = get_model(args.base_model, True, model_name=args.model_name)
    model.eval()

    # Load annotated data
    KTDataset = KTDatasetPacked if PACK_KCS else KTDatasetUnpacked
    KTCollator = KTCollatorPacked if PACK_KCS else KTCollatorUnpacked
    get_loss = get_loss_packed if PACK_KCS else get_loss_unpacked
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    test_dataset = KTDataset(test_df, tokenizer, args)
    collator = KTCollator(tokenizer)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Collect meta data and predicted KC/correctness probabilities for test set
    dialogue_idx_to_sample_idxs = {}
    all_labels = []
    all_preds = []
    all_kc_probs = []
    all_kcs = []
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        # TODO: potentially evaluate on all KCs in dialogue not just ones relevant for each turn
        for sample_idx, sample in enumerate(batch["meta_data"]):
            dialogue_idx_to_sample_idxs.setdefault(sample["dialogue_idx"], []).append(batch_idx + sample_idx)
        with torch.no_grad():
            loss, kc_probs, corr_probs = get_loss(model, batch, true_token, false_token)
        total_loss += loss.item()
        all_labels.extend(batch["labels"].tolist())
        all_preds.extend(corr_probs.tolist())
        all_kc_probs.extend(kc_probs)
        all_kcs.extend([sample["kcs"] for sample in batch["meta_data"]])

    # Compute quantitative metrics across all turns and only on final turns
    loss = total_loss / len(test_dataloader)
    result_str = f"Loss: {loss:.4f}\n"
    result_str += f"Overall ({len(all_labels)} samples):\n"
    result_str += f"GT - True: {sum(all_labels)}, False: {len(all_labels) - sum(all_labels)}; "
    result_str += f"Pred - True: {sum(np.round(all_preds))}, False: {len(all_preds) - sum(np.round(all_preds))}\n"
    all_metrics = compute_metrics(all_labels, all_preds)
    result_str += "Acc: {:.4f}, AUC: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}\n".format(*all_metrics)
    final_turn_labels = [all_labels[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    final_turn_preds = [all_preds[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    result_str += f"Final Turn ({len(final_turn_labels)} samples):\n"
    result_str += f"GT - True: {sum(final_turn_labels)}, False: {len(final_turn_labels) - sum(final_turn_labels)}; "
    result_str += f"Pred - True: {sum(np.round(final_turn_preds))}, False: {len(final_turn_preds) - sum(np.round(final_turn_preds))}\n"
    final_metrics = compute_metrics(final_turn_labels, final_turn_preds)
    result_str += "Acc: {:.4f}, AUC: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}\n".format(*final_metrics)
    print(result_str)
    with open(f"results/metrics_{args.model_name}.txt", "w") as out_file:
        out_file.write(result_str)

    # Save KCs and probability predictions to file for analysis
    kc_results = {
        dialogue_idx: [
            {
                kc: kc_prob
                for kc, kc_prob in zip(all_kcs[sample_idx], all_kc_probs[sample_idx])
            }
            for sample_idx in sample_idxs
        ]
        for dialogue_idx, sample_idxs in dialogue_idx_to_sample_idxs.items()
    }
    with open(get_kc_result_filename(args), "w") as out_file:
        json.dump(kc_results, out_file, indent=2)

    return [loss, *all_metrics, *final_metrics]
