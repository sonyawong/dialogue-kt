import json
import matplotlib.pyplot as plt

from data_loading import get_kc_result_filename

def visualize_single(kc_info):
    target_dialogue_idx = list(kc_info.keys())[3] # 3, tried up to 9
    dialogue = kc_info[target_dialogue_idx]
    kc_to_curve = {}
    for turn_idx, turn in enumerate(dialogue):
        for kc, prob in turn.items():
            kc_to_curve.setdefault(kc, {"x": [], "y": []})
            kc_to_curve[kc]["x"].append(turn_idx)
            kc_to_curve[kc]["y"].append(prob)
    plt.rcParams["figure.figsize"] = (12,8)
    for kc, curve in kc_to_curve.items():
        plt.plot(curve["x"], curve["y"], label=kc)
    plt.legend()
    plt.show()

def visualize_average(kc_info):
    # TODO: find most commonly occurring KCs in the data (by number of dialogues they appear in)
    # TODO: for each of those KCs, find average delta after 1, 2, 3, etc. turns
    # TODO: alternatively and more simply, if collecting predictions for all kcs at each turn, show average delta from first turn at each subsequent turn
    # TODO: plot the kcs and the average deltas
    pass

def visualize(args):
    with open(get_kc_result_filename(args)) as kc_file:
        kc_info = json.load(kc_file)

    # visualize_single(kc_info)
    visualize_average(kc_info)
