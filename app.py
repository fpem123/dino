import argparse
import json
import re
import traceback
import time
import os
from typing import Any, Dict
from queue import Queue, Empty
from threading import Thread

from modeling import GPT2Wrapper, DinoGenerator, PLACEHOLDER_STR
from utils import set_seed, read_inputs, DatasetEntry
from flask import Flask, request, render_template, jsonify
import torch

cuda_device = True if torch.cuda.is_available() else False

app = Flask(__name__)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1


##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    requests["output"] = gen_dino(requests['input'][0])
                except Exception as e:
                    traceback.print_exc()
                    requests["output"] = 'error'


handler = Thread(target=handle_requests_by_batch).start()

def validate_args(args) -> None:
    """Validate the given command line arguments"""
    if args.input_file is not None:
        assert args.num_entries_per_input_and_label is not None, "If 'input_file' is set, 'num_entries_per_input_and_label' must be set"
        assert args.num_entries_per_label is None, "If 'input_file' is set, 'num_entries_per_label' must not be set"
    else:
        assert args.num_entries_per_input_and_label is None, "If 'input_file' is not set, 'num_entries_per_input_and_label' must not be set"
        assert args.num_entries_per_label is not None, "If 'input_file' is not set, 'num_entries_per_label' must be set"


def validate_task_spec(task_spec: Dict[str, Any], with_inputs: bool) -> None:
    """Validate the given task specification"""
    error_prefix = "Invalid task specification:"
    assert 'task_name' in task_spec, f"{error_prefix} missing field 'task_name'"
    assert isinstance(task_spec['task_name'], str) and re.match(r"^[A-Za-z0-9\-_.]+$", task_spec['task_name']), \
        f"{error_prefix} 'task_name' must be a string consisting only of [A-Za-z0-9\\-_.]"
    assert 'labels' in task_spec, f"{error_prefix} missing field 'labels'"
    assert isinstance(task_spec['labels'], dict), f"{error_prefix} 'labels' must be a dictionary"
    all_labels = task_spec['labels'].keys()
    for label, label_dict in task_spec['labels'].items():
        assert isinstance(label_dict, dict), f"{error_prefix} label '{label}' is not mapped to a dictionary"
        assert 'instruction' in label_dict.keys(), f"{error_prefix} missing field 'instruction' for label '{label}'"
        assert 'counter_labels' in label_dict.keys(), f"{error_prefix} missing field 'counter_labels' for label '{label}'"
        assert isinstance(label_dict['instruction'], str), f"{error_prefix} 'instruction' not a string for label '{label}'"
        assert isinstance(label_dict['counter_labels'], list), f"{error_prefix} 'counter_labels' not a list for label '{label}'"
        assert label_dict['instruction'][-1] == '"', \
            f"{error_prefix} each instruction should end with an opening quotation mark (\") so that the next quotation mark generated " \
            f"by the model can be interpreted as a signal that it is done."
        if with_inputs:
            assert label_dict['instruction'].count(PLACEHOLDER_STR) == 1, \
                f"{error_prefix} The instruction for label '{label}' does not contain exactly one placeholder token ({PLACEHOLDER_STR}). " \
                f"If an input file is specified, each instruction must contain this placeholder to indicate where the input should be " \
                f"inserted."
        else:
            assert label_dict['instruction'].count(PLACEHOLDER_STR) == 0, \
                f"{error_prefix} The instruction for label '{label}' contains a placeholder token ({PLACEHOLDER_STR}). If no input file " \
                f"is specified, instructions must not contain this placeholder as there is no input to replace it with."
        for counter_label in label_dict['counter_labels']:
            assert counter_label in all_labels, f"{error_prefix} counter_label '{counter_label}' for label '{label}' is not a label"


##
# GPT-2 generator.
def gen_dino(text):
    try:
        dino_input = text.splitlines()
        outputs = generator.generate_dataset(dino_input,
                                             num_entries_per_input_and_label=args.num_entries_per_input_and_label,
                                             num_entries_per_label=args.num_entries_per_label)

        result = dict()

        for i, output in enumerate(outputs):
            result[i] = dict()
            result[i]['text_a'] = output.text_a
            result[i]['text_b'] = output.text_b
            result[i]['label'] = output.label

        return result

    except Exception as e:
        traceback.print_exc()
        return {'error': e}, 500


##
# Get post request page.
@app.route('/gen', methods=['POST'])
def generate():
    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return {'Error': 'Too Many Requests'}, 429

    try:
        text = request.form['text']

        args = [text]

    except Exception as e:
        traceback.print_exc()
        return {'message': 'Invalid request'}, 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    return result, 200


##
# Main page.
@app.route('/')
def web_main():
    return render_template('main.html'), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--task_file", type=str, required=True,
                        help="A json file providing the instructions and other information required for dataset generation. "
                             "See the 'task_specs' directory for examples and 'README.md' for more details on how to create this file.")

    # Text generation and sampling parameters
    parser.add_argument("--model_name", type=str, default="skt/kogpt2-base-v2",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")
    parser.add_argument("--max_output_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--decay_constant", type=float, default=100,
                        help="The decay constant for self-debiasing")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")

    # Dataset parameters
    parser.add_argument("--input_file", type=str,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")
    parser.add_argument("--input_file_type", choices=["plain", "jsonl", "stsb"], default="plain",
                        help="The type of the input file. Choices are 'plain' (a raw text file with one input per line), 'jsonl' (a jsonl "
                             "file as produced by DINO) and 'stsb' (a TSV file in the STS Benchmark format)")
    parser.add_argument("--num_entries_per_input_and_label", type=int, default=None,
                        help="The number of entries to generate for each pair of input text and label (only if --input_file is set)")
    parser.add_argument("--num_entries_per_label", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--remove_duplicates", action='store_true',
                        help="Whether duplicates should be removed from the generated dataset")
    parser.add_argument("--remove_identical_pairs", action='store_true',
                        help="Whether text pairs with text_a == text_b should be removed from the dataset (only for text pair datasets)")
    parser.add_argument("--keep_outputs_without_eos", action='store_true',
                        help="If set to true, examples where the language model does not output a quotation mark (which is interpreted as "
                             "a signal that it has completed its output) are not removed from the dataset.")
    parser.add_argument("--allow_newlines_in_outputs", action='store_true',
                        help="If set to true, model outputs that contain a newline character before the end-of-sequence token (a quotation "
                             "mark) are not removed from the dataset.")
    parser.add_argument("--min_num_words", type=int, default=-1,
                        help="The minimum number of (whitespace-separated) words for each dataset entry. Entries with fewer words are "
                             "removed.")

    # Miscellaneous further parameters
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    validate_args(args)

    set_seed(args.seed)

    with open(args.task_file, 'r', encoding='utf8') as fh:
        task_specification = json.load(fh)
        validate_task_spec(task_specification, with_inputs=args.input_file is not None)

    wrapper = GPT2Wrapper(model_name="skt/kogpt2-base-v2", use_cuda=cuda_device)
    generator = DinoGenerator(
        model=wrapper,
        task_spec=task_specification,
        max_output_length=args.max_output_length,
        decay_constant=args.decay_constant,
        top_p=args.top_p,
        top_k=args.top_k,
        remove_duplicates=args.remove_duplicates,
        remove_identical_pairs=args.remove_identical_pairs,
        min_num_words=args.min_num_words,
        keep_outputs_without_eos=args.keep_outputs_without_eos,
        allow_newlines_in_outputs=args.allow_newlines_in_outputs
    )

    from waitress import serve
    serve(app, port=80, host='0.0.0.0')
