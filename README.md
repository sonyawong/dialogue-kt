# Dialogue Knowledge Tracing
This repository contains the code for the paper <a href="">Exploring Knowledge Tracing in Tutor-Student Dialogues</a>. The primary contributions here include the code for the LLMKT and DKT-Sem models, the code for running deep KT and BKT models on dialogue KT, and the code for automatically annotating dialogues with KT labels using the OpenAI API.

If you use our code or find this work useful in your research then please cite us!
```
TODO
```

## Setup

### Download Data
<b>Achieve the Core (ATC)</b>: Download the <a href="https://huggingface.co/datasets/allenai/achieve-the-core">ATC HuggingFace dataset</a> and put `standards.jsonl` and `domain_groups.json` under `src/ATC`. At the time of releasing this code, the data was not accessible via HuggingFace due to a bug. If the data is still not accessible then you can contact us or the authors of <a href="https://arxiv.org/pdf/2408.04226">the paper</a> to send you a copy.

<b>CoMTA</b>: Download the <a href="https://github.com/Khan/tutoring-accuracy-dataset/blob/main/CoMTA_dataset.json">CoMTA data file</a> and put it under `data/src`.

<b>MathDial</b>: Clone the <a href="https://github.com/eth-nlped/mathdial/tree/main">MathDial repo</a> and put the root under `data/src`.

### Environment
We used Python 3.10.12 in the development of this work. Run the following to set up a Python environment:
```
python -m venv dk
source dk/bin/activate
pip install -r requirements.txt
```

Also add the following to your environment:
```
export OPENAI_API_KEY=<your key here> # For automated annotation via OpenAI
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # For enabling deterministic operations
```

## Prepare Dialogues for KT (Run Annotation with OpenAI)
Dialogue KT requires each dialogue turn to be annotated with correctness and KC labels. We automated this process with LLM prompting via the OpenAI API. You can run the following to tag correctness and ATC standard KCs on the two datasets:
```
python main.py annotate --mode collect --openai_model gpt-4o --dataset comta
python main.py annotate --mode collect --openai_model gpt-4o --dataset mathdial
```

To see statistics on the resulting labels, run:
```
python main.py annotate --mode analyze --dataset comta
python main.py annotate --mode analyze --dataset mathdial
```

## Evaluate KT Methods
Each of the following runs a train/test cross-validation on the CoMTA data for a different model:
```
python main.py train --dataset comta --crossval --model_type lmkt --model_name lmkt_model         # LLMKT
python main.py train --dataset comta --crossval --model_type dkt-sem --model_name dkt_sem_model   # DKT-Sem
python main.py train --dataset comta --crossval --model_type dkt --model_name dkt_model           # DKT
python main.py train --dataset comta --crossval --model_type dkvmn --model_name dkvmn_model       # DKVMN
python main.py train --dataset comta --crossval --model_type akt --model_name akt_model           # AKT
python main.py train --dataset comta --crossval --model_type saint --model_name saint_model       # SAINT
python main.py train --dataset comta --crossval --model_type simplekt --model_name simplekt_model # simpleKT
python main.py train --dataset comta --crossval --model_type bkt                                  # BKT
```

Check the `results` folder for metric summaries and turn-level predictions for analysis.

To see all training options, run:
```
python main.py train --help
```

### Hyperparameter Sweep
We run a grid search to find the optimal hyperparameters for the DKT family models. For example, to run a search for DKT on CoMTA, run the following (crossval is inferred and model_name is set automatically):
```
python main.py train --dataset comta --hyperparam_sweep --model_type dkt
```

The output will indicate the model that achieved the highest validation AUC. To get its performance on the test folds, run:
```
python main.py test --dataset comta --crossval --model_type dkt --model_name <copy from output> --emb_size <get from model_name>
```

#### Best Hyperparameters Found

CoMTA:
- DKT-Sem: lr=5e-3, emb_size=128
- DKT: lr=5e-3, emb_size=64
- DKVMN: lr=2e-4, emb_size=8
- AKT: lr=1e-3, emb_size=16
- SAINT: lr=1e-4, emb_size=8
- simpleKT: lr=1e-3, emb_size=32

MathDial:
- DKT-Sem: lr=1e-3, emb_size=256
- DKT: lr=5e-4, emb_size=256
- DKVMN: lr=5e-3, emb_size=32
- AKT: lr=2e-4, emb_size=256
- SAINT: lr=1e-3, emb_size=64
- simpleKT: lr=5e-4, emb_size=256

## Visualize Learning Curves
To generate the learning curve graphs, run the following (they will be placed in `results`):
```
python main.py visualize --dataset comta --model_name <trained model to visualize predictions for>
```
