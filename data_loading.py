from typing import List
import json
import re
from ast import literal_eval
import pandas as pd

def add_content(cur: str, new: str):
    new = new.strip()
    if not cur:
        return new
    if cur == new: # Sometimes turns are repeated in MathDial
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new

def process_dialogue(turns: List[dict]):
    cur_role = turns[0]["role"]
    cur_turn = {
        "turn": 0 if cur_role == "student" else 1,
        "teacher": "",
        "student": ""
    }
    result = []
    for turn in turns:
        if turn["role"] == "teacher" and cur_role == "student":
            result.append(cur_turn)
            cur_turn = {
                "turn": cur_turn["turn"] + 1,
                "teacher": "",
                "student": ""
            }
        cur_role = turn["role"]
        cur_turn[cur_role] = add_content(cur_turn[cur_role], turn["content"])
    # Only include final turn if there was a student response
    if cur_turn["student"]:
        result.append(cur_turn)
    return result

def load_comta_src_data():
    with open("data/src/CoMTA_dataset.json") as file:
        data = json.load(file)
    proc_data = []
    for sample in data:
        # Skip calculus since not in ATC
        if sample["math_level"] == "Calculus":
            continue
        # Add dialogue and meta data
        proc_data.append({
            "dialogue": process_dialogue([
                {"role": "student" if turn["role"] == "user" else "teacher", "content": turn["content"]}
                for turn in sample["data"]
            ]),
            "meta_data": {
                "expected_result": sample["expected_result"],
                "math_level": sample["math_level"]
            }
        })
    return pd.DataFrame(proc_data)

def load_mathdial_src_data(split: str):
    turn_prefix_re = re.compile(r"^[a-zA-Z]+: (\([a-z]+\))?")
    with open(f"data/src/mathdial/data/{split}.jsonl") as file:
        data = [json.loads(line) for line in file]
    proc_data = []
    for sample in data:
        # Skip dialogues that are rated as not being typical of student behavior
        if (not sample["self-typical-confusion"] or not sample["self-typical-interactions"]
            or sample["self-typical-confusion"] < 4 or sample["self-typical-interactions"] < 4):
            continue
        # Add dialogue and meta data
        proc_data.append({
            "dialogue": process_dialogue([
                {"role": "teacher" if turn.startswith("Teacher") else "student", "content": turn_prefix_re.sub("", turn)}
                for turn in sample["conversation"].split("|EOM|")
            ]),
            "meta_data": {
                "question": sample["question"],
                "correct_solution": sample["ground_truth"],
                "incorrect_solution": sample["student_incorrect_solution"],
                "self_correctness": sample["self-correctness"]
            }
        })
    return pd.DataFrame(proc_data)

def load_src_data(args, split: str = ""):
    if args.dataset == "comta":
        return load_comta_src_data()
    elif args.dataset == "mathdial":
        return load_mathdial_src_data(split)
    raise Exception(f"Loading not supported for {args.dataset}")

def get_annotated_data_filename(args, split: str = ""):
    return f"data/annotated/{args.dataset}{f'_{split}' if split else ''}_{args.tag_src}.csv"

def load_annotated_data(args, fold: int = 0):
    if args.dataset == "comta":
        df = pd.read_csv(get_annotated_data_filename(args), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        df = df.sample(frac=1, random_state=221)
        split_point = int(len(df) * (fold / 5))
        df = pd.concat([df[split_point:], df[:split_point]])
        return (
            df[:int(len(df) * .65)],
            df[int(len(df) * .65) : int(len(df) * .8)],
            df[int(len(df) * .8):],
        )
    elif args.dataset == "mathdial":
        train_df = pd.read_csv(get_annotated_data_filename(args, "train"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        train_df = train_df.sample(frac=1, random_state=221)
        test_df = pd.read_csv(get_annotated_data_filename(args, "test"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        return (
            train_df[:int(.8 * len(train_df))],
            train_df[int(.8 * len(train_df)):],
            test_df
        )
    raise Exception(f"Loading not supported for {args.dataset}")

def get_kc_result_filename(args):
    model_name = args.model_name or args.base_model.replace("/", "-")
    return f"results/kcs_{args.dataset}_{model_name}.json"

def load_atc():
    with open("data/src/ATC/domain_groups.json") as file:
        domain_groups = json.load(file)

    with open("data/src/ATC/standards.jsonl") as file:
        standards = [json.loads(line) for line in file]

    for stand in standards:
        stand["description"] = stand["description"].split("\nGrade")[0] # Remove grade-level descriptions
        stand["description"] = stand["description"].replace("\n", " ") # Remove newlines for easier LM prompting

    return {
        "domain_groups": domain_groups,
        "standards": {tag["id"]: tag for tag in standards}
    }
