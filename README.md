# Vera

This repo hosts the code for [Vera: A General-Purpose Plausibility Estimation Model for Commonsense Statements](https://arxiv.org/abs/2305.03695)

If you are interested in using the Vera model, please visit our demo or download the model from HuggingFace.
This repo is mainly for model training and reproducing the results in the paper.

* Model: <https://huggingface.co/liujch1998/vera>
* Demo: <https://huggingface.co/spaces/liujch1998/vera>

## Setup

Create a conda environment and activate it:
```bash
conda env create -f environment.yml
conda activate vera
```

## Prepare Data

The training and evaluation data should be in the format of declarative statements.
Each dataset split should be a JSON file with the following format:
```json
[
    {
        "golds": [
            "Green is a color."
        ],
        "distractors": [
            "Sky is a color.",
            "Bread is a color."
        ]
    },
    {
        "golds": [
            "Green is a color."
        ],
        "distractors": []
    },
    {
        "golds": [],
        "distractors": [
            "Sky is a color."
        ]
    }
]
```
The JSON file should contain a list of problems, and each problem is a dictionary with correct statements under `golds` and incorrect statements under `distractors`.
* The first problem examplifies a multiple-choice problem, which should have one correct statement and one or more incorrect statements.
* The second is a boolean problem with label True, which should have one correct statement and no incorrect statement.
* The third is a boolean problem with label False, which should have no correct statement and one incorrect statement.

In practice, each JSON file should contain either purely multiple-choice problems or purely boolean problems.

To prepare the datasets for training,
1. Download the question converter model (Chen et al., 2021) by following [this link](https://github.com/jifan-chen/QA-Verification-Via-NLI/blob/master/seq2seq_converter/run_question_converter.sh#L5). Put the model ckpt at `ckpt/question-converter-t5-3b/`
1. Download the datasets. The source from which we retrieved the datasets can be found in the Appendix of our paper.
1. Convert each dataset into a TSV file in [UnifiedQA format](https://github.com/allenai/unifiedqa#feeding-data-into-unifiedqa) (if it is not already in that format).
1. For each dataset, create a directory for it under `data/`, and put the training and dev sets in this directory.
    * For example, for CommonsenseQA you can create `data/commonsenseqa/train.tsv` and `data/commonsenseqa/dev.tsv`
1. Run `python prepare_data.py {dataset} {split}` to convert a dataset split into declarative format.
    * For example, to convert the dev set of CommonsenseQA, run `python prepare_data.py commonsenseqa dev`. The output file can be found at `data/commonsenseqa/dev.json`
    * To batch prepare the datasets used in the Vera paper, you may run `sh prepare_data.sh`
    * If you're using a new dataset that is not in the Vera paper, you might need to slightly modify `prepare_data.py` to fit the new dataset.
1. Register your datasets in `datasets.py`. Datasets used in the Vera paper are already registered, so it'd be most convenient to follow the naming conventions indicated there.

For evaluation datasets, follow the same procedure above, except that you only need to prepare the evaluation split of each dataset.

Note: The above procedure does not apply to the following datasets, and you need to prepare them separately:
* `atomic2020_3d` and `genericskb_3d`
* `symkd_anno`, `gengen_anno` and `rainier_anno`

## Training

To train a commonsense verification model based on the T5 Encoder on 1 GPU, run
```bash
accelerate launch \
    run.py \
    --train_tasks {dataset1,dataset2,...} --valid_tasks {dataset1,dataset2,...} \
    --run_name "train"
```
which by default would set the base model to be the encoder of T5-v1.1-small, and the per-GPU batch size to be 1.
Refer to `run.py` for the list of customizable parameters.

To replicate the Vera model trained in the paper, run the following two commands sequentially:
```bash
# Stage A training
accelerate launch \
    --num_processes 64 \
    --mixed_precision bf16 \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_backward_prefetch_policy BACKWARD_PRE --fsdp_offload_params false --fsdp_sharding_strategy 1 --fsdp_state_dict_type FULL_STATE_DICT --fsdp_transformer_layer_cls_to_wrap T5Block \
    run.py \
    --model_type google/t5-v1_1-xxl \
    --run_name "train_stage_a"

# Stage B training
accelerate launch \
    --num_processes 64 \
    --mixed_precision bf16 \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_backward_prefetch_policy BACKWARD_PRE --fsdp_offload_params false --fsdp_sharding_strategy 1 --fsdp_state_dict_type FULL_STATE_DICT --fsdp_transformer_layer_cls_to_wrap T5Block \
    run.py \
    --model_type google/t5-v1_1-xxl \
    --load_from_ckpt {PATH_TO_STAGE_A_CKPT} \
    --run_name "train_stage_b"
```
where `{PATH_TO_STAGE_A_CKPT}` is the path to the model ckpt from Stage A training, and it should be something like `../runs/train_stage_a/model/ckp_XXXXX.pth`

## Evaluation

To evaluate a trained model, run
```bash
accelerate launch \
    run.py \
    --mode eval \
    --load_from_ckpt {PATH_TO_CKPT} \
    --eval_tasks {dataset1,dataset2,...} \
    --run_name "eval"
```
where `{PATH_TO_CKPT}` is the path to the model ckpt from training, and it should be something like `../runs/train/model/ckp_XXXXX.pth`

To evaluate the replicated Vera model trained in the previous section, run
```bash
accelerate launch \
    --num_processes 64 \
    --mixed_precision bf16 \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_backward_prefetch_policy BACKWARD_PRE --fsdp_offload_params false --fsdp_sharding_strategy 1 --fsdp_state_dict_type FULL_STATE_DICT --fsdp_transformer_layer_cls_to_wrap T5Block \
    run.py \
    --mode eval \
    --model_type google/t5-v1_1-xxl \
    --load_from_ckpt {PATH_TO_STAGE_B_CKPT} \
    --run_name "eval_stage_b"
```

## Citation

If you find this repo useful, please consider citing our paper:
```bibtex
@article{Liu2023VeraAG,
  title={Vera: A General-Purpose Plausibility Estimation Model for Commonsense Statements},
  author={Jiacheng Liu and Wenya Wang and Dianzhuo Wang and Noah A. Smith and Yejin Choi and Hanna Hajishirzi},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.03695}
}
```
