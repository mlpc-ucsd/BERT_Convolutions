# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""


import dataclasses
import logging
import os
import copy
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from src.transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from src.transformers import GlueDataTrainingArguments as DataTrainingArguments
from src.transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    hyperparams: Optional[str] = field(
        default="electra", metadata={"help": "What hyperparameters to use: electra, electra_base, stable"}
    )


# The task to be run should be in the data_args.task_name field.
def run_task(model_args, data_args, training_args):
    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if not training_args.do_train:
        model_args.model_name_or_path = training_args.output_dir

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if data_args.task_name == 'mnli': # Save a copy of the tuned subgraph for MNLI, to use for other tasks.
            trainer.model.bert.save_pretrained(os.path.join(training_args.output_dir, "transformer_subgraph"))
            print("Saved transformer subgraph tuned on MNLI.")
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )
            diagnostic_data_args = dataclasses.replace(data_args, task_name="diagnostic")
            test_datasets.append(
                GlueDataset(diagnostic_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def run_all_tasks(model_args, data_args, training_args, dir_suffix=""):
    # Set seed
    set_seed(training_args.seed)

    # Run the task(s).
    glue_data_dir = data_args.data_dir
    output_dir = training_args.output_dir
    model_dir = model_args.model_name_or_path

    # Treat the batch_size flag as the max batch size. May be lowered for some tasks.
    max_per_device_batch_size = training_args.per_device_train_batch_size
    for task_name in ['MNLI', 'CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'QNLI', 'RTE', 'WNLI']:
        data_args.task_name = task_name.lower()
        data_args.data_dir = os.path.join(glue_data_dir, task_name)
        training_args.output_dir = os.path.join(output_dir, task_name)
        training_args.output_dir = training_args.output_dir + dir_suffix

        # Overwrite default ELECTRA hyperparameters.
        if "electra" in model_args.hyperparams:
            print("Using hyperparameters from ELECTRA.")
            # Approximate ELECTRA steps (10 epochs for RTE and  STS, 3 epochs for others), with batch size 32:
            electra_steps = {'MNLI': 36364, 'CoLA': 851, 'SST-2': 6409, 'MRPC': 343, 'STS-B': 1836,
                             'QQP': 33600, 'QNLI': 9835, 'RTE': 777, 'WNLI': 100} # 5 epochs for WNLI.
            training_args.learning_rate = 0.0003
            training_args.per_device_train_batch_size = 32 // training_args.n_gpu
            training_args.max_steps = electra_steps[task_name]
            training_args.warmup_steps = electra_steps[task_name] // 10
            # Different from ELECTRA paper: use a larger batch size for MNLI and QQP, as in ALBERT:
            if task_name=="MNLI" or task_name=="QQP":
                training_args.per_device_train_batch_size *= 4
                # New MNLI steps: ~12121
                # New QQP steps: ~11200
                # x(4/3)//4 because training for 4 epochs instead of 3, but using 4x larger batches.
                training_args.max_steps = training_args.max_steps // 3
                training_args.warmup_steps = training_args.max_steps // 10
            # To improve QNLI stability, lower learning rate:
            if task_name=="QNLI":
                training_args.learning_rate = 0.0001
        if model_args.hyperparams == "electra_base":
            # Use electra hyperparameters but smaller learning rate for base size model.
            print("Using LR 0.0001.")
            training_args.learning_rate = 0.0001
        if model_args.hyperparams == "stable":
            # This is too slow for larger datasets.
            # Approximate steps for one epoch using batch size 16.
            epoch_steps = {'MNLI': 24243, 'CoLA': 567, 'SST-2': 4273, 'MRPC': 229, 'STS-B': 367,
                           'QQP': 22400, 'QNLI': 6557, 'RTE': 155, 'WNLI': 100}
            training_args.learning_rate = 0.00002
            training_args.per_device_train_batch_size = 16 // training_args.n_gpu
            training_args.max_steps = epoch_steps[task_name]*20
            training_args.warmup_steps = epoch_steps[task_name]*2
        # Use the MNLI checkpoint for RTE, STS, and MRPC (as in the ALBERT paper).
        model_args.model_name_or_path = model_dir
        if task_name=="RTE" or task_name=="STS-B" or task_name=="MRPC":
            # Assume MNLI has already been run.
            model_args.model_name_or_path = os.path.join(output_dir, "MNLI" + dir_suffix + "/transformer_subgraph")

        if training_args.per_device_train_batch_size > max_per_device_batch_size:
            training_args.per_device_train_batch_size = max_per_device_batch_size
        training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size

        print('\nRUNNING TASK: {}\n'.format(task_name))
        run_task(model_args, data_args, training_args)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    if data_args.task_name == 'all' or data_args.task_name == 'ALL':
        # For this case, run ten sets of finetuning.
        suffixes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        seeds = [42, 97, 834, 622, 329, 75, 213, 101, 123, 57] # Default seed is 42.
        if training_args.seed != 42:
            seeds = [seed + training_args.seed for seed in seeds]
        for i in range(len(suffixes)):
            training_args.seed = seeds[i]
            run_all_tasks(copy.deepcopy(model_args),
                          copy.deepcopy(data_args),
                          copy.deepcopy(training_args),
                          dir_suffix=suffixes[i])
    else:
        run_task(model_args, data_args, training_args)
    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
