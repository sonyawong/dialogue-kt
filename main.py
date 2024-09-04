import argparse

from annotate import annotate
from training import train, test
from visualize import visualize
from utils import initialize_seeds

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_annotate = subparsers.add_parser("annotate", help="Annotate dialogues")
    parser_annotate.set_defaults(func=annotate)
    parser_annotate.add_argument("--mode", type=str, choices=["collect", "analyze"])
    parser_annotate.add_argument("--use_azure", action="store_true")
    parser_annotate.add_argument("--openai_model", type=str)

    parser_train = subparsers.add_parser("train", help="Train KT model")
    parser_train.set_defaults(func=train)
    parser_train.add_argument("--epochs", type=int, default=5)
    parser_train.add_argument("--lr", type=float, default=2e-4)
    parser_train.add_argument("--wd", type=float, default=1e-2)
    parser_train.add_argument("--gc", type=float, default=1.0)
    parser_train.add_argument("--grad_accum_steps", type=int, default=8)
    parser_train.add_argument("--r", type=int, default=16)
    parser_train.add_argument("--lora_alpha", type=int, default=16)
    parser_train.add_argument("--crossval", action="store_true")

    parser_test = subparsers.add_parser("test", help="Test KT model")
    parser_test.set_defaults(func=test)

    parser_visualize = subparsers.add_parser("visualize", help="Visualize KCs")
    parser_visualize.set_defaults(func=visualize)

    for subparser in [parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
        subparser.add_argument("--model_name", type=str)
        subparser.add_argument("--batch_size", type=int, default=8)
        subparser.add_argument("--testonval", action="store_true")

    for subparser in [parser_annotate, parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--dataset", type=str, choices=["comta", "mathdial"], default="comta")
        subparser.add_argument("--tag_src", type=str, choices=["base", "atc"], default="atc")
        subparser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
