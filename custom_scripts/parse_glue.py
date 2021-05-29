"""
Parse eval GLUE scores from a directory.
"""

from __future__ import unicode_literals

import codecs
import argparse
import os
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--test_dir', default='')
    return parser

def main(input, test_dir):
    suffixes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    score_dicts = [{} for i in range(len(suffixes))]
    tasks = [("CoLA", "eval_results_cola.txt", "eval_mcc"),
             ("MNLI-m", "eval_results_mnli.txt", "eval_mnli/acc"),
             ("MNLI-mm", "eval_results_mnli-mm.txt", "eval_mnli-mm/acc"),
             ("MRPC", "eval_results_mrpc.txt", "eval_acc_and_f1"),
             ("QNLI", "eval_results_qnli.txt", "eval_acc"),
             ("QQP", "eval_results_qqp.txt", "eval_acc_and_f1"),
             ("RTE", "eval_results_rte.txt", "eval_acc"),
             ("SST-2", "eval_results_sst-2.txt", "eval_acc"),
             ("STS-B", "eval_results_sts-b.txt", "eval_corr"),
             ("WNLI", "eval_results_wnli.txt", "eval_acc")]
    for i in range(len(suffixes)):
        for task, eval_path, task_metric in tasks:
            task_dir_name = "MNLI" if "MNLI" in task else task
            full_eval_path = os.path.join(input, task_dir_name)
            full_eval_path = full_eval_path + suffixes[i]
            full_eval_path = os.path.join(full_eval_path, eval_path)
            try:
                infile = codecs.open(full_eval_path, 'rb', encoding='utf-8')
                for line in infile:
                    if task_metric in line:
                        score = float(line.split()[-1])
                        score = score*100
                        score_dicts[i][task] = score
                        break
                infile.close()
            except FileNotFoundError:
                score_dicts[i][task] = 'N/A'
    # Print results.
    print("GLUE EVAL SCORES:")
    for task, _ in score_dicts[0].items():
        print(task, end="\t")
    print('', end="\n")
    for score_dict in score_dicts:
        for _, score in score_dict.items():
            print(round(score, 1), end="\t")
        print('', end="\n")

    # Compile test set results.
    if test_dir == "":
        return
    for task, eval_path, _ in tasks:
        scores = [score_dict[task] for score_dict in score_dicts]
        best_index = np.argmax(scores)
        task_dir_name = "MNLI" if "MNLI" in task else task
        full_eval_path = os.path.join(input, task_dir_name)
        full_eval_path = full_eval_path + suffixes[best_index]
        full_eval_path = os.path.join(full_eval_path, eval_path)

        full_test_path = full_eval_path.replace("eval", "test")
        print("Using results: {}".format(full_test_path))
        infile = codecs.open(full_test_path, 'rb', encoding='utf-8')
        lines = infile.readlines()
        infile.close()
        # For STS-B, the values need to be between 0 and 5.
        if task == "STS-B":
            fixed_lines = []
            for line_count, line in enumerate(lines):
                if line_count == 0:
                    fixed_lines.append(line)
                    continue
                split = line.strip().split()
                score = float(split[1])
                if score < 0:
                    score = 0.000
                if score > 5:
                    score = 5.000
                split[1] = str(score)
                fixed_lines.append("{}\n".format("\t".join(split)))
            lines = fixed_lines
        output_path = os.path.join(test_dir, "{}.tsv".format(task))
        outfile = codecs.open(output_path, 'w', encoding='utf-8')
        for line in lines:
            outfile.write(line)
        outfile.close()

        # Diagnostic file.
        if task == "MNLI-m":
            diagnostic_path = full_test_path.replace("test_results_mnli.txt", "test_results_diagnostic.txt")
            infile = codecs.open(diagnostic_path, 'rb', encoding='utf-8')
            lines = infile.readlines()
            infile.close()
            output_path = os.path.join(test_dir, "AX.tsv")
            outfile = codecs.open(output_path, 'w', encoding='utf-8')
            for line in lines:
                outfile.write(line)
            outfile.close()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input, args.test_dir)
