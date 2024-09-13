from typing import List, Set, Dict
import json
import re
import numpy as np
import pandas as pd

from openai_api import OpenAIClient
from data_loading import load_src_data, get_annotated_data_filename, get_kc_dict_filename, load_annotated_data, load_atc
from prompting import anno_base_system_prompt, anno_base_user_prompt, anno_atc_system_prompt, anno_atc_user_prompt, anno_correctness_system_prompt, get_dialogue_text
from kt_data_loading import apply_annotations

def extract_result(annotation: str):
    annotation = annotation.replace("\\(", "").replace("\\)", "").replace("\\pi", "pi") # LaTeX (in base tagging) causes JSON error
    match = re.match(r".*(```json(.*)```|result = (.*))", annotation, re.DOTALL)
    if not match:
        return None
    try:
        anno_json = json.loads(match.group(2) or match.group(3))
    except json.decoder.JSONDecodeError:
        return None
    if "result" in anno_json:
        anno_json = anno_json["result"]
    return anno_json

def combine_kcs_and_correctness(data: pd.DataFrame, kcs: List[dict], correctness: List[dict], atc: dict = None):
    num_failed = 0
    annotations = []
    for (_, sample), dia_kcs, dia_corrs in zip(data.iterrows(), kcs, correctness):
        if dia_kcs is None or dia_corrs is None:
            num_failed += 1
            annotations.append({"error": "JSON parsing error"})
        elif len(dia_kcs) != len(dia_corrs):
            num_failed += 1
            annotations.append({"error": "Different number of standard and correctness output turns"})
        elif len(sample["dialogue"]) != len(dia_corrs) + (1 if sample["dialogue"][0]["turn"] == 0 else 0):
            num_failed += 1
            annotations.append({"error": "Different number of annotations and source turns"})
        else:
            dia_anno = {}
            for idx in range(len(dia_corrs)):
                key = f"turn {idx + 1}"
                turn_kcs = dia_kcs[key]
                turn_corr = dia_corrs[key]
                # Convert correctness string to True/False/None, override to None if no KCs for that turn
                turn_corr = None if turn_corr == "na" or not turn_kcs else True if turn_corr == "true" else False
                # Override KCs with empty list if correctness is None
                turn_kcs = [] if turn_corr is None else turn_kcs
                # If doing ATC tagging, convert IDs to strings
                if atc is not None:
                    turn_kcs = [atc["standards"][tag_id]["description"] for tag_id in turn_kcs]
                # Save current turn
                dia_anno[key] = {"correct": turn_corr, "kcs": turn_kcs}
            annotations.append(dia_anno)
    print(f"Num failed: {num_failed} / {len(data)}")
    return annotations

def create_kc_dict(df: pd.DataFrame):
    kc_dict = {}
    for _, sample in df.iterrows():
        if "error" in sample["annotation"]:
            continue
        for turn in sample["annotation"].values():
            for kc in turn["kcs"]:
                if kc not in kc_dict:
                    kc_dict[kc] = len(kc_dict)
    return kc_dict

def collect_base(args, split):
    data = load_src_data(args, split)
    if args.debug:
        data = data[:2]
    client = OpenAIClient(args.use_azure)

    # Tag knowledge components
    print("Tagging knowledge components...")
    prompts = [anno_base_user_prompt(sample, args) for _, sample in data.iterrows()]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_base_system_prompt(args), show_progress=True)
    kcs = [extract_result(result) for result in results]
    data["kc_annotation_raw"] = results
    data["kc_annotation"] = kcs

    # Tag correctness
    print("Tagging correctness...")
    prompts = [anno_base_user_prompt(sample, args) for _, sample in data.iterrows()]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_correctness_system_prompt(args), show_progress=True)
    correctness = [extract_result(res) for res in results]
    data["correctness_annotation_raw"] = results
    data["correctness_annotation"] = correctness

    # Validate/process annotations and save to output file
    data["annotation"] = combine_kcs_and_correctness(data, kcs, correctness)
    data.to_csv(get_annotated_data_filename(args, split), index=False)
    return data

def get_atc_options(parent_ids: List[str], level: str, atc: dict):
    assert level in ("cluster", "standard")
    if level == "cluster":
        # Collect meta ids that are associated with the selected domains
        domain_ids = [dom_id for dom_name in parent_ids for dom_id in atc["domain_groups"][dom_name]["domain_cats"]]
        # Collect true ids associated with meta ids (either grade.id for K-8 or parent = id for high school)
        parent_ids = [
            dom["id"] for dom in atc["standards"].values()
            if dom["level"] == "Domain" and any([
                dom["id"].endswith(f".{dom_id}") or dom["parent"] == dom_id for dom_id in domain_ids
            ])
        ]
    # Get all children of selected parents, format option strings and sort
    parent_ids = list(set(parent_ids)) # Remove any duplicates
    options = [atc["standards"][tag] for par_id in parent_ids for tag in atc["standards"][par_id]["children"]]
    options = sorted([f"ID: {tag['id']}, Description: {tag['description']}" for tag in options])
    return options

def collect_atc(args, split: str):
    data = load_src_data(args, split)
    if args.debug:
        data = data[:2]
    atc = load_atc()
    client = OpenAIClient(args.use_azure)

    # Tag domains
    print("Tagging domains...")
    domain_options = [
        f"Name: {name}, Description: {dom['description']}"
        for name, dom in atc["domain_groups"].items()
    ]
    prompts = [anno_atc_user_prompt(sample, "domain", domain_options, args) for _, sample in data.iterrows()]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_atc_system_prompt("domain", args), show_progress=True)
    domains = [extract_result(res) for res in results]
    data["domain_annotation_raw"] = results
    data["domain_annotation"] = domains

    # Tag clusters
    print("Tagging clusters...")
    valid_idxs = [idx for idx, dom in enumerate(domains) if dom is not None]
    print(f"Num valid idxs: {len(valid_idxs)} / {len(data)}")
    cluster_options = [get_atc_options(domains[idx], "cluster", atc) for idx in valid_idxs]
    prompts = [anno_atc_user_prompt(data.iloc[idx], "cluster", opts, args) for idx, opts in zip(valid_idxs, cluster_options)]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_atc_system_prompt("cluster", args), show_progress=True)
    results_inc_none = [None] * len(data)
    clusters = [None] * len(data)
    for idx, val_idx in enumerate(valid_idxs):
        results_inc_none[val_idx] = results[idx]
        clusters[val_idx] = extract_result(results[idx])
    data["cluster_annotation_raw"] = results_inc_none
    data["cluster_annotation"] = clusters

    # Tag standards
    print("Tagging standards...")
    valid_idxs = [idx for idx, clust in enumerate(clusters) if clust is not None]
    print(f"Num valid idxs: {len(valid_idxs)} / {len(data)}")
    standard_options = [get_atc_options(clusters[idx], "standard", atc) for idx in valid_idxs]
    prompts = [anno_atc_user_prompt(data.iloc[idx], "standard", opts, args) for idx, opts in zip(valid_idxs, standard_options)]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_atc_system_prompt("standard", args), show_progress=True)
    results_inc_none = [None] * len(data)
    standards = [None] * len(data)
    for idx, val_idx in enumerate(valid_idxs):
        results_inc_none[val_idx] = results[idx]
        standards[val_idx] = extract_result(results[idx])
    data["standard_annotation_raw"] = results_inc_none
    data["standard_annotation"] = standards

    # Tag correctness
    print("Tagging correctness...")
    prompts = [anno_base_user_prompt(sample, args) for _, sample in data.iterrows()]
    results = client.get_batched_responses(prompts, args.openai_model, 4000, 10, 0,
                                           system_message=anno_correctness_system_prompt(args), show_progress=True)
    correctness = [extract_result(res) for res in results]
    data["correctness_annotation_raw"] = results
    data["correctness_annotation"] = correctness

    # Validate/process annotations and save to output file
    data["annotation"] = combine_kcs_and_correctness(data, standards, correctness, atc)
    data.to_csv(get_annotated_data_filename(args, split), index=False)
    return data

def collect(args, split: str = ""):
    assert args.openai_model
    if args.tag_src == "atc":
        return collect_atc(args, split)
    return collect_base(args, split)

def analyze(args):
    train_df, val_df, test_df = load_annotated_data(args)
    data = pd.concat([train_df, val_df, test_df])
    num_turns: List[int] = []
    num_correct: List[int] = []
    num_na: List[int] = []
    final_correct_match: List[bool] = []
    per_dia_num_kcs: List[int] = []
    per_turn_num_kcs: List[int] = []
    all_kcs: Set[str] = set()
    parse_failed: List[int] = []
    subject_to_count: Dict[str, int] = {}
    for idx, sample in data.iterrows():
        dialogue = sample["dialogue"]
        anno = sample["annotation"]
        dialogue_anno = apply_annotations(sample)
        if "error" in anno:
            parse_failed.append(idx)
            continue
        anno = list(anno.values())
        num_turns.append(dialogue[-1]["turn"])
        num_correct.append(len([0 for turn in anno if turn["correct"]]))
        num_na.append(len([0 for turn in anno if turn["correct"] is None]))
        final_correct_match.append(anno[-1]["correct"] == dialogue_anno[-1]["correct"])
        kc_set = {kc for turn in anno for kc in turn["kcs"]}
        per_dia_num_kcs.append(len(kc_set))
        per_turn_num_kcs.extend([len(turn["kcs"]) for turn in anno if turn["kcs"]])
        all_kcs = all_kcs.union(kc_set)
        if args.dataset == "comta":
            subject_to_count.setdefault(sample["meta_data"]["math_level"], 0)
            subject_to_count[sample["meta_data"]["math_level"]] += 1
    total_num_turns = sum(num_turns)
    # print("All KCs:\n" + "\n".join(sorted(all_kcs)))
    if subject_to_count:
        print("Subject Counts - " + ", ".join([f"{subject}: {count}" for subject, count in subject_to_count.items()]))
    print(f"Turns - Total: {total_num_turns}, Avg: {total_num_turns / len(data):.4f}")
    print(f"Correct - True: {sum(num_correct) / total_num_turns:.4f}, "
          f"False: {(total_num_turns - sum(num_correct) - sum(num_na)) / total_num_turns:.4f}, "
          f"NA: {sum(num_na) / total_num_turns:.4f}")
    print(f"Final Correct Match: {sum(final_correct_match) / len(data):.4f}")
    print(f"Num KCs - Total: {len(all_kcs)}, Avg per Dialogue: {sum(per_dia_num_kcs) / len(data):.4f}, Avg per Turn: {np.mean(per_turn_num_kcs)}")
    print(f"Parsing Failed: {len(parse_failed)} / {len(data)} ({parse_failed})")

def create_human_annotation_files(args):
    train_df, val_df, test_df = load_annotated_data(args)
    data = pd.concat([train_df, val_df, test_df])

    # Tutor Question, Student Response, Predicted Correctness, Correctness Accuracy, Predicted Standards, Standards Rating

def annotate(args):
    if args.mode == "llm-collect":
        # Collect and save dialogue annotations
        if args.dataset == "mathdial":
            print("Annotating train split...")
            train_data = collect(args, "train")
            print("Annotating test split...")
            test_data = collect(args, "test")
            data = pd.concat([train_data, test_data])
        else:
            data = collect(args)
        # Save resulting KC dictionary
        kc_dict = create_kc_dict(data)
        with open(get_kc_dict_filename(args), "w") as file:
            json.dump(kc_dict, file, indent=2, ensure_ascii=False)
    elif args.mode == "llm-analyze":
        analyze(args)
    elif args.mode == "human-create":
        create_human_annotation_files(args)
    elif args.mode == "human-analyze":
        pass
