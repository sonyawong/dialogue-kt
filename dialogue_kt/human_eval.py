from itertools import combinations
import pandas as pd
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.stats import pearsonr

from dialogue_kt.data_loading import load_annotated_data, correct_to_str, standards_to_str, COMTA_SUBJECTS
from dialogue_kt.kt_data_loading import apply_annotations

def create_human_annotation_files(args):
    train_df, val_df, test_df = load_annotated_data(args)
    data = pd.concat([train_df, val_df, test_df]).sample(n=30, random_state=221)
    results = []
    for idx, sample in data.iterrows():
        dialogue = apply_annotations(sample, apply_na=False)
        for turn in dialogue:
            results.append({
                "Dialogue ID": idx + 1,
                "Turn": turn["turn"],
                "Tutor Question": turn["teacher"] or "--",
                "Student Response": turn["student"],
                "Predicted Correctness": correct_to_str(turn["og_correct"]),
                "Correctness Accuracy": "1" if turn["turn"] == 0 else "",
                "Predicted Standards": standards_to_str(turn["kcs"], "\n"),
                "Standards Rating": "4" if turn["turn"] == 0 else ""
            })
        results.append({key: "" for key in results[0]}) # Add empty row between dialogues
    pd.DataFrame(results).to_csv(f"data/annotated/{args.dataset}_human_eval.csv", index=False)

def analyze_human_annotation_files(args):
    filenames = ["data/annotated/comta_human_eval_resp_1.csv", "data/annotated/comta_human_eval_resp_2.csv", "data/annotated/comta_human_eval_resp_3.csv"]
    correctness_scores = []
    correctness_scores_no_final = []
    kc_scores = []
    kc_scores_no_final = []
    kc_scores_by_subject = {subj: [] for subj in COMTA_SUBJECTS}
    train_df, val_df, test_df = load_annotated_data(args)
    src_df = pd.concat([train_df, val_df, test_df]).sample(n=30, random_state=221)
    src_df = src_df.sort_index()
    src_df["Dialogue ID"] = src_df.index + 1
    final_turn_gt_correctness = src_df.apply(lambda row: row["meta_data"]["expected_result"] == "Answer Accepted", axis=1)
    rater_accs = []
    print("Prediction Accuracy:")
    for rater_idx, filename in enumerate(filenames):
        # Load and clean annotation data
        df = pd.read_csv(filename)
        df = df[df["Dialogue ID"].notna()]
        df = df[df["Turn"] > 0]
        df = df.sort_values(["Dialogue ID", "Turn"])
        df = df.merge(src_df[["Dialogue ID", "meta_data"]], on="Dialogue ID")
        df.loc[df["Correctness Accuracy"] == "na", "Correctness Accuracy"] = 0
        df.loc[df["Standards Rating"].isna(), "Standards Rating"] = 1
        df["subject"] = df.apply(lambda row: row["meta_data"]["math_level"], axis=1)
        for subj in COMTA_SUBJECTS:
            kc_scores_by_subject[subj].extend(df[df["subject"] == subj]["Standards Rating"].tolist())
        # Collect ratings
        correctness_scores.append(df["Correctness Accuracy"].astype(float).tolist())
        kc_scores.append(df["Standards Rating"].astype(float).tolist())
        # Collect ratings excluding final turn
        dia_group = df.groupby("Dialogue ID", sort=False)
        all_but_last = dia_group.apply(lambda group: group.iloc[:-1])
        correctness_scores_no_final.append(all_but_last["Correctness Accuracy"].astype(float).tolist())
        kc_scores_no_final.append(all_but_last["Standards Rating"].astype(float).tolist())
        # Test rater accuracy on final turn
        last_corr_anno = dia_group["Correctness Accuracy"].last().astype(bool)
        last_ai_corr_pred = dia_group["Predicted Correctness"].last().apply(lambda val: val == "TRUE")
        rater_corr_pred = ~(last_corr_anno ^ last_ai_corr_pred)
        rater_acc = (rater_corr_pred.tolist() == final_turn_gt_correctness).sum() / len(rater_corr_pred)
        rater_accs.append(rater_acc)
        print(f"Rater {rater_idx + 1} Accuracy: {rater_acc:.4f}")

    print(f"Avg. Rater Accuracy: ${np.mean(rater_accs):.4f} \\pm {np.std(rater_accs):.4f}$")
    ai_accuracy = (last_ai_corr_pred.tolist() == final_turn_gt_correctness).sum() / len(last_ai_corr_pred)
    print(f"AI Accuracy: {ai_accuracy:.4f}")
    print(f"Total Num. Labels: {len(correctness_scores[0])}")

    # Report average ratings and inter-rater reliability metrics across labels
    for name, scores in [
        ("Correctness", correctness_scores), ("KCs", kc_scores),
        ("Correctness (No Final)", correctness_scores_no_final), ("KCs (No Final)", kc_scores_no_final)
    ]:
        print(f"\n{name}:")
        scores = np.array(scores)
        print("Proportions - " + ", ".join([f"{val}: {(scores == val).mean():.4f}" for val in range(int(scores.min()), int(scores.max()) + 1)]))
        avg_score = scores.mean()
        print(f"Avg. Score - Overall: {avg_score:.4f}, " + ", ".join([f"{idx + 1}: {np.mean(scores_cur):.4f}" for idx, scores_cur in enumerate(scores)]))
        alpha = krippendorff.alpha(reliability_data=scores)
        acc = np.prod([scores[0] == scores_cur for scores_cur in scores[1:]], axis=0).mean()
        print(f"Krippendorff Alpha: {alpha:.4f}, Overlap: {acc:.4f}")
        for idx0, idx1 in combinations(range(0, len(scores)), 2):
            acc = accuracy_score(scores[idx0], scores[idx1])
            corr = pearsonr(scores[idx0], scores[idx1]).statistic
            kappa = cohen_kappa_score(scores[idx0], scores[idx1], weights="quadratic")
            print(f"IRR ({idx0 + 1}, {idx1 + 1}) - Acc: {acc:.4f}, Correlation: {corr:.4f}, QWK: {kappa:.4f}")

    print("\nPer-Subject KC Scores: " + ", ".join([f"{subj}: {np.mean(kc_scores_by_subject[subj]):.4f}" for subj in COMTA_SUBJECTS]))

def human_eval(args):
    if args.mode == "create":
        create_human_annotation_files(args)
    elif args.mode == "analyze":
        analyze_human_annotation_files(args)
