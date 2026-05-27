# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained SCBench helpers for benchmark_scbench.py.

Replaces the former runtime dependency on FastKVZip's ``prefill`` package so
the benchmark runs without cloning that repo. The dataset loading, prompt
templates, generation-length presets, and answer metrics here are adapted
from FastKVZip (https://github.com/Janghyun1230/FastKVzip), which in turn
adapts the SCBench metrics from Microsoft MInference
(https://github.com/microsoft/MInference/tree/main/scbench).

Provides:
    load_dataset_all, get_data_list, template, set_gen_length,
    evaluate_answer, f1_score
"""

import re
import string
from collections import Counter

# ---------------------------------------------------------------------------
# Dataset name groups (FastKVZip/prefill/eval.py: get_data_list)
# ---------------------------------------------------------------------------

_SHORT = ["squad", "gsm"]
_MID = [
    "scbench_many_shot",
    "scbench_mf",
    "scbench_choice_eng",
    "scbench_qa_eng",
    "scbench_repoqa",
]
_LONG = [
    "scbench_kv",
    "scbench_prefix_suffix",
    "scbench_summary",
    "scbench_vt",
]
_MULTI = [
    "scbench_summary_with_needles",
    "scbench_repoqa_and_kv",
]


def get_data_list(dataname: str, modelname: str = "") -> list[str]:
    """Expand a group name (short/mid/long/multi/all) into dataset names.

    A non-group name is returned as a single-element list. For qwen3/gemma3
    models, certain tasks fall back to their shortened variants."""
    if dataname == "short":
        data_list = list(_SHORT)
    elif dataname == "mid":
        data_list = list(_MID)
    elif dataname == "long":
        data_list = list(_LONG)
    elif dataname == "multi":
        data_list = list(_MULTI)
    elif dataname == "all":
        data_list = _LONG + _SHORT + _MID
    else:
        data_list = [dataname]

    if any(k in modelname.lower() for k in ("qwen3", "gemma3", "gemma-3")):
        data_list = [
            f"{x}_short" if x == "scbench_prefix_suffix" else x
            for x in data_list
        ]
        if "instruct" not in modelname.lower():
            data_list = [
                f"{x}_short" if x == "scbench_kv" else x for x in data_list
            ]
            data_list = [
                f"{x}_mid" if x == "scbench_mf" else x for x in data_list
            ]

    print(data_list)
    return data_list


# ---------------------------------------------------------------------------
# Prompt templates (FastKVZip/prefill/model/template.py: template)
# ---------------------------------------------------------------------------

def template(model_name: str, task: str) -> tuple[str, str]:
    """Return (prefix, postfix) wrapping a context+question prompt for the
    given model family and task."""
    model_name = model_name.lower()

    if "llama" in model_name or model_name == "duo":
        prefix = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )
        postfix = (
            "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_name.startswith("qwen"):
        prefix = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        if "qwen3-" in model_name and "instruct" not in model_name:
            postfix += "<think>\n\n</think>\n\n"
    elif model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
        prefix = "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n"
        postfix = "<end_of_turn>\n<start_of_turn>model\n"
    else:
        print(
            "**Warning** No prompt template for this model; using a generic "
            "fallback (see scbench_local.template)."
        )
        prefix = "<|begin_of_text|>"
        postfix = "\n\nAnswer: "

    if task.startswith("gsm"):
        prefix += (
            "Given the context, answer to the following reasoning "
            "question.\n\n"
        )
    else:
        prefix += (
            "Given the context, answer to the following question or request "
            "without explanation.\n\n"
        )

    return prefix, postfix


# ---------------------------------------------------------------------------
# Generation length presets (FastKVZip/prefill/utils/func.py: set_gen_length)
# ---------------------------------------------------------------------------

def set_gen_length(dataname: str) -> int:
    """Default max output tokens per dataset."""
    if any(k in dataname for k in ("needle", "_mf")):
        max_len = 48
    elif "prefix_suffix" in dataname:
        max_len = 128
    elif any(k in dataname for k in ("squad", "summary")):
        max_len = 256
    elif any(k in dataname for k in ("gsm", "repoqa")):
        max_len = 512
    else:
        max_len = 96
    print(f"set generation length: {max_len} (see scbench_local.set_gen_length)")
    return max_len


# ---------------------------------------------------------------------------
# Dataset loading (FastKVZip/prefill/data/load.py: load_dataset_all)
# ---------------------------------------------------------------------------

def _check_scbench_name(name: str) -> None:
    tag = name.split("scbench_")[1]
    possible_tags = [
        "many_shot", "mf", "repoqa", "choice_eng", "prefix_suffix",
        "summary", "qa_eng", "vt", "kv", "summary_with_needles",
        "repoqa_and_kv",
    ]
    for suffix in ("_tiny", "_short", "_mid"):
        if suffix.strip("_") in tag:
            tag = tag.split(suffix)[0]
            break
    assert tag in possible_tags, f"SCBench data name does not exist: {name!r}"


def _load_scbench(name: str) -> list[dict]:
    from datasets import load_dataset

    _check_scbench_name(name)
    samples = load_dataset(
        "Jang-Hyun/SCBench-preprocessed",
        data_files=f"{name}.parquet",
        split="train",
    )

    dataset = []
    for data in samples:
        d = {"context": data["prompts"][0], "question": data["prompts"][1:]}
        answers = []
        for gt in data["ground_truth"]:
            answers.append(", ".join(gt) if isinstance(gt, list) else str(gt))
        d["answers"] = answers
        dataset.append(d)
    return dataset


def _load_squad(n_data: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset("rajpurkar/squad", split="train")
    pool: dict[str, int] = {}
    contexts: list[str] = []
    questions: list[list[str]] = []
    answers: list[list[str]] = []
    for d in data:
        ctx = d["context"]
        if ctx not in pool:
            pool[ctx] = len(contexts)
            contexts.append(ctx)
            questions.append([d["question"]])
            answers.append(list(d["answers"]["text"]))
        else:
            idx = pool[ctx]
            questions[idx].append(d["question"])
            answers[idx].append(d["answers"]["text"][0])
        if len(pool) > n_data:
            break
    return [
        {"context": c, "question": q, "answers": a}
        for c, q, a in zip(contexts, questions, answers)
    ]


def _load_gsm(tokenizer, n_data: int) -> list[dict]:
    from datasets import load_dataset

    dataset_full = load_dataset("openai/gsm8k", "main", split="test")
    dataset = []
    for data in dataset_full:
        st = data["question"].split(". ")
        context = ". ".join(st[:-1]).strip() + "."
        if len(tokenizer.encode(context, add_special_tokens=False)) < 72:
            continue
        dataset.append(
            {
                "context": context,
                "question": [st[-1].strip()],
                "answers": [data["answer"]],
            }
        )
        if len(dataset) == n_data:
            break
    return dataset


def load_dataset_all(name: str, tokenizer, n_data: int = 100) -> list[dict]:
    """Load a dataset as a list of {context, question[list], answers[list]}."""
    if name == "squad":
        dataset = _load_squad(n_data)
    elif name == "gsm":
        dataset = _load_gsm(tokenizer, n_data)
    elif "scbench" in name:
        dataset = _load_scbench(name)
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}. Supported: scbench_*, squad, gsm."
        )
    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


# ---------------------------------------------------------------------------
# Answer metrics (FastKVZip/prefill/results/metric.py, from MInference SCBench)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def replace_num(text):
        word_to_number = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        }
        pattern = re.compile(
            r"\b(" + "|".join(word_to_number.keys()) + r")\b"
        )
        return pattern.sub(lambda x: word_to_number[x.group()], text)

    return replace_num(
        white_space_fix(remove_articles(remove_punc(s.lower())))
    )


def f1_score(pred: str, ref: str, normalize: bool = True) -> float:
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    prediction_tokens = pred.split()
    ground_truth_tokens = ref.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


_ROUGE_SCORER = None


def _rouge_score(prediction: str, ground_truth: str) -> float:
    # Lazy: only summary-family tasks need ROUGE. Uses google-research's
    # ``rouge-score`` (the package vLLM already pins in requirements/test.txt),
    # not the similarly named ``rouge`` package.
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    try:
        # RougeScorer.score(target, prediction) — target is the reference.
        return _ROUGE_SCORER.score(ground_truth, prediction)["rougeL"].fmeasure
    except Exception:
        return 0.0


def _include_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _include_score_multi(pred, ref, normalize=True):
    refs = ref.split(", ")
    if normalize:
        pred = normalize_answer(pred)
        refs = [normalize_answer(r) for r in refs]
    scores = [r in pred for r in refs]
    return sum(scores) / len(scores)


def _include_score_gsm(pred, ref, normalize=True):
    ref = ref.strip().split("#### ")[-1]
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _include_score_manyshot(pred, ref, normalize=True):
    if "(" in pred and "(" in ref:
        pred = pred.split("(")[1].split(")")[0]
        ref = ref.split("(")[1].split(")")[0]
        return pred == ref
    if ref and ref[0] == "(":
        ref = ref.split(")")[1].strip()
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def _exact_match_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return pred == ref


def evaluate_answer(
    preds, refs, dataname, fmt, similarity=False, subtask=None
) -> list[float]:
    """Score predictions against references with the SCBench per-task metric.

    The RepoQA structured-similarity path (repo_qa_utils) is intentionally not
    bundled; callers should pass ``similarity=True`` for repoqa datasets (as
    benchmark_scbench.py does), which routes to token-level F1."""
    if "repoqa" in dataname and not similarity:
        raise NotImplementedError(
            "RepoQA structured scoring is not bundled in scbench_local; pass "
            "similarity=True to score repoqa via F1."
        )

    score: list[float] = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if pred.endswith("</s>"):
            pred = pred[:-4]
        if len(pred.strip()) == 0:
            score.append(0.0)
            continue

        name = subtask[i] if subtask is not None else dataname

        if similarity:
            score.append(f1_score(pred, ref))
        elif fmt != "qa":
            score.append(_rouge_score(pred, ref))
        elif "_vt" in name:
            score.append(_include_score_multi(pred, ref, normalize=False))
        elif "_mf" in name:
            score.append(_exact_match_score(pred, ref, normalize=False))
        elif "_many_shot" in name:
            score.append(_include_score_manyshot(pred, ref))
        elif "summary" in name:
            score.append(_rouge_score(pred, ref))
        elif "qa_eng" in name:
            score.append(max(f1_score(pred, ref), _include_score(pred, ref)))
        elif "choice_eng" in name:
            score.append(_include_score(pred.split("\n")[0], ref))
        elif "gsm" in name:
            pred = pred.strip().lower().split("the answer is ")[-1]
            score.append(_include_score_gsm(pred, ref, normalize=False))
        else:
            score.append(_include_score(pred, ref))
    return [float(s) for s in score]
