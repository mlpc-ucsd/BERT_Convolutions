# BERT Convolutions
Code for the paper [Convolutions and Self-Attention: Re-interpreting Relative Positions in Pre-trained Language Models](https://arxiv.org/abs/2106.05505).
Contains experiments for integrating convolutions and self-attention in BERT models.
Code is adapted from [Huggingface Transformers](https://github.com/huggingface/transformers).
Model code is in src/transformers/modeling_bert.py.
Run on Python 3.6.9 and Pytorch 1.7.1 (see requirements.txt).

## Training
To train tokenizer, use custom_scripts/train_spm_tokenizer.py.
To pre-train BERT with a plain text dataset:
<pre>
python3 run_language_modeling.py \
--model_type=bert \
--tokenizer_name="./data/sentencepiece/spm.model" \
--config_name="./data/bert_base_config.json" \
--do_train --mlm --line_by_line \
--train_data_file="./data/training_text.txt" \
--per_device_train_batch_size=32 \
--save_steps=25000 \
--block_size=128 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./bert-experiments/bert"
</pre>

The code above produces a cached file of examples (a list of lists of token indices).
Each example is an un-truncated and un-padded sentence pair (but includes \[CLS\] and \[SEP\] tokens).
Convert these lists to an iterable text file using custom_scripts/shuffle_cached_dataset.py.
Then, you can pre-train BERT using an iterable dataset (saving memory):
<pre>
python3 run_language_modeling.py \
--model_type=bert \
--tokenizer_name="./data/sentencepiece/spm.model" \
--config_name="./data/bert_base_config.json" \
--do_train --mlm --train_iterable --line_by_line \
--train_data_file="./data/iterable_pairs_train.txt" \
--per_device_train_batch_size=32 \
--save_steps=25000 \
--block_size=128 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./bert-experiments/bert"
</pre>

Optional flags to change BERT architecture when pre-training from scratch:<br/>
In the following, qk uses query/key self-attention, convfixed is a fixed lightweight convolution, convq is query-based dynamic lightweight convolution (relative embeddings), convk is a key-based dynamic lightweight convolution, and convolution is a fixed depthwise convolution.
<pre>--attention_kernel="qk_convfixed_convq_convk [num_positions_each_dir]"</pre>
Remove absolute position embeddings:
<pre>--remove_position_embeddings</pre>
Convolutional values, using depthwise-separable (depth) convolutions for half of heads (mixed), and using no activation function (no_act) between the depthwise and pointwise convolutions:
<pre>--value_forward="convolution_depth_mixed_no_act [num_positions_each_dir] [num_convolution_groups]"</pre>
Convolutional queries/keys for half of heads:
<pre>--qk="convolution_depth_mixed_no_act [num_positions_each_dir] [num_convolution_groups]"</pre>

## Fine-tuning
Training and evaluation for downstream GLUE tasks (note: batch size represents max batch size, because batch size is adjusted for each task):
<pre>
python3 run_glue.py \
--data_dir="./glue-data/data-tsv" \
--task_name=ALL \
--save_steps=9999999 \
--max_seq_length 128 \
--per_device_train_batch_size 99999 \
--tokenizer_name="./data/sentencepiece/spm.model" \
--model_name_or_path="./bert-experiments/bert" \
--output_dir="./bert-experiments/bert-glue" \
--hyperparams="electra_base" \
--do_eval \
--do_train
</pre>

## Prediction
Run the fine-tuned models on the GLUE test set:<br/>
This adds a file with test set predictions to each GLUE task directory.
<pre>
python3 run_glue.py \
--data_dir="./glue-data/data-tsv" \
--task_name=ALL \
--save_steps=9999999 \
--max_seq_length 128 \
--per_device_train_batch_size 99999 \
--tokenizer_name="./data/sentencepiece/spm.model" \
--model_name_or_path="./bert-experiments/placeholder" \
--output_dir="./bert-experiments/bert-glue" \
--hyperparams="electra_base" \
--do_predict
</pre>
Then, test results can be compiled into one directory.
The test_results directory will contain test predictions, using the fine-tuned model with the highest dev set score for each task.
The files in test_results can be zipped and submitted to the GLUE benchmark site for evaluation.
<pre>
python3 custom_scripts/parse_glue.py \
--input="./bert-experiments/bert-glue" \
--test_dir="./bert-experiments/bert-glue/test_results"
</pre>

## Citation
<pre>
@inproceedings{chang-etal-2021-convolutions,
  title={Convolutions and Self-Attention: Re-interpreting Relative Positions in Pre-trained Language Models},
  author={Tyler Chang and Yifan Xu and Weijian Xu and Zhuowen Tu},
  booktitle={ACL-IJCNLP 2021},
  year={2021},
}
</pre>
