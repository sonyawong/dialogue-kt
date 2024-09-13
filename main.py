import argparse

from annotate import annotate
from training import train, test, BASELINE_MODELS
from visualize import visualize
from utils import initialize_seeds, bool_type

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_annotate = subparsers.add_parser("annotate", help="Annotate dialogues")
    parser_annotate.set_defaults(func=annotate)
    parser_annotate.add_argument("--mode", type=str, choices=["llm-collect", "llm-analyze", "human-create", "human-analyze"])
    parser_annotate.add_argument("--use_azure", action="store_true")
    parser_annotate.add_argument("--openai_model", type=str)

    parser_train = subparsers.add_parser("train", help="Train KT model")
    parser_train.set_defaults(func=train)
    parser_train.add_argument("--epochs", type=int)
    parser_train.add_argument("--lr", type=float)
    parser_train.add_argument("--wd", type=float)
    parser_train.add_argument("--gc", type=float)
    parser_train.add_argument("--grad_accum_steps", type=int)
    parser_train.add_argument("--r", type=int)
    parser_train.add_argument("--lora_alpha", type=int)
    parser_train.add_argument("--optim", type=str, choices=["adamw", "adafactor"], default="adamw")
    parser_train.add_argument("--pt_model_name", type=str)
    parser_train.add_argument("--hyperparam_sweep", action="store_true")

    parser_test = subparsers.add_parser("test", help="Test KT model")
    parser_test.set_defaults(func=test)

    parser_visualize = subparsers.add_parser("visualize", help="Visualize KCs")
    parser_visualize.set_defaults(func=visualize)

    for subparser in [parser_annotate, parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--dataset", type=str, choices=["comta", "mathdial"], default="comta")
        subparser.add_argument("--split_by_subject", action="store_true")
        subparser.add_argument("--typical_cutoff", type=int, default=1)
        subparser.add_argument("--tag_src", type=str, choices=["base", "atc"], default="atc")
        subparser.add_argument("--debug", action="store_true")

    for subparser in [parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
        subparser.add_argument("--model_name", type=str)

    for subparser in [parser_train, parser_test]:
        subparser.add_argument("--model_type", type=str, choices=["lmkt"] + BASELINE_MODELS + ["random", "majority"], default="lmkt")
        subparser.add_argument("--batch_size", type=int)
        subparser.add_argument("--crossval", action="store_true")
        subparser.add_argument("--testonval", action="store_true")
        subparser.add_argument("--pack_kcs", type=bool_type, default=True)
        subparser.add_argument("--quantize", type=bool_type, default=False)
        subparser.add_argument("--prompt_inc_labels", type=bool_type, default=False)
        subparser.add_argument("--emb_size", type=int)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
