"""
evaluate.py
Standalone evaluation script for RAG on FEVER fact verification.
Loads best checkpoint, runs full test set evaluation, and saves results.

Reports:
    - 3-way label accuracy (SUPPORTS / REFUTES / NOT ENOUGH INFO)
    - 2-way label accuracy (SUPPORTS / REFUTES only, excluding NEI)
    - Per-class precision, recall, F1 via sklearn classification_report
    - Confusion matrix
    - Qualitative examples (correct and incorrect per label)

Final results from reproduction run (3 epochs, full 228K training set):
    3-way accuracy: 66.2%  (paper: 72.5%)
    2-way accuracy: 75.1%  (paper: 89.5%)

See DECISIONS.md for explanation of the gap to the paper.
"""

import os
import sys
import json
import time
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data import (
    LABEL2ID, ID2LABEL,
    load_fever, load_passages,
    apply_faiss_setup, move_index_to_gpu,
    path_setup
)
from model import (
    load_models_from_checkpoint,
    search_index, forward_pass,
    predict, LABEL_TOKEN_IDS
)


def run_evaluation(test_data, model, q_encoder, bart_tokenizer,
                   q_tokenizer, gpu_index, passages,
                   max_length=300, n_docs=5):
    """
    Run inference on the full test set and collect predictions.

    Args:
        test_data (list):   list of FEVER test examples
        model:              BartForConditionalGeneration in eval mode
        q_encoder:          DPRQuestionEncoder in eval mode
        bart_tokenizer:     BartTokenizer
        q_tokenizer:        DPRQuestionEncoderTokenizerFast
        gpu_index:          FAISS GPU index
        passages (list):    list of passage dicts
        max_length (int):   max source token length. Defaults to 300.
        n_docs (int):       number of passages to retrieve. Defaults to 5.

    Returns:
        gold_labels (list): gold label strings
        pred_labels (list): predicted label strings
        all_probs (list):   softmax probabilities [n_examples, 3]
    """
    gold_labels = []
    pred_labels = []
    all_probs   = []
    t_start     = time.time()

    print(f"Running evaluation on {len(test_data):,} examples...")

    for i, example in enumerate(test_data):
        if i % 1000 == 0:
            elapsed   = time.time() - t_start
            remaining = (elapsed / max(i, 1)) * (len(test_data) - i)
            print(f"  {i:,}/{len(test_data):,} | "
                  f"elapsed: {elapsed/60:.1f}min | "
                  f"remaining: {remaining/60:.1f}min")

        results = search_index(
            example["claim"], gpu_index, passages,
            q_encoder, q_tokenizer,
            n_docs=n_docs,
            max_length=max_length,
            training=False
        )[0]

        marginalized = forward_pass(
            example["claim"], results, example["label"],
            model, bart_tokenizer, LABEL2ID,
            max_length=max_length
        )[0]

        probs      = torch.softmax(marginalized, dim=0)
        pred_label = predict(marginalized, LABEL2ID)

        gold_labels.append(example["label"])
        pred_labels.append(pred_label)
        all_probs.append(probs.cpu().numpy())

    elapsed = time.time() - t_start
    print(f"\nEvaluation complete in {elapsed/60:.1f} minutes")

    return gold_labels, pred_labels, all_probs


def compute_metrics(gold_labels, pred_labels):
    """
    Compute 3-way accuracy, 2-way accuracy, and per-class report.

    Args:
        gold_labels (list): gold label strings
        pred_labels (list): predicted label strings

    Returns:
        acc_3way (float): 3-way accuracy
        acc_2way (float): 2-way accuracy (SUPPORTS/REFUTES only)
        report (str):     sklearn classification report string
    """
    # 3-way accuracy
    acc_3way = accuracy_score(gold_labels, pred_labels)

    # 2-way accuracy — filter out NEI
    pairs_2way   = [
        (g, p) for g, p in zip(gold_labels, pred_labels)
        if g != "NOT ENOUGH INFO"
    ]
    gold_2way    = [g for g, p in pairs_2way]
    pred_2way    = [p for g, p in pairs_2way]
    acc_2way     = accuracy_score(gold_2way, pred_2way)

    report = classification_report(
        gold_labels, pred_labels,
        target_names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        digits=3
    )

    return acc_3way, acc_2way, report


def plot_results(gold_labels, pred_labels, acc_3way, acc_2way,
                 results_dir):
    """
    Generate and save confusion matrix and accuracy comparison plots.

    Args:
        gold_labels (list):  gold label strings
        pred_labels (list):  predicted label strings
        acc_3way (float):    3-way accuracy
        acc_2way (float):    2-way accuracy
        results_dir (str):   directory to save plots
    """
    labels_order = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    cm = confusion_matrix(
        gold_labels, pred_labels, labels=labels_order
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # confusion matrix
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=["SUP", "REF", "NEI"],
        yticklabels=["SUP", "REF", "NEI"],
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Gold")
    axes[0].set_title("Confusion matrix — test set")

    # accuracy vs paper
    categories  = ["3-way accuracy", "2-way accuracy"]
    ours_vals   = [acc_3way, acc_2way]
    paper_vals  = [0.725, 0.895]

    x = np.arange(len(categories))
    w = 0.35
    bars = axes[1].bar(
        x - w/2, ours_vals, w, label="Ours",  color="#3498db"
    )
    axes[1].bar(
        x + w/2, paper_vals, w, label="Paper",
        color="#e67e22", alpha=0.7
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Our results vs paper (RAG-Token)")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    for bar, val in zip(bars, ours_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", fontsize=9
        )

    plt.tight_layout()
    save_path = os.path.join(results_dir, "final_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {save_path}")


def save_results(acc_3way, acc_2way, gold_labels, pred_labels,
                 results_dir):
    """
    Save final results summary to JSON.

    Args:
        acc_3way (float):   3-way accuracy
        acc_2way (float):   2-way accuracy
        gold_labels (list): gold label strings
        pred_labels (list): predicted label strings
        results_dir (str):  directory to save results
    """
    results = {
        "model": "RAG (facebook/rag-token-nq fine-tuned on FEVER)",
        "checkpoint": "best_gpu (epoch 3)",
        "test_set": "copenlu/fever_gold_evidence test split",
        "n_test_examples": len(gold_labels),
        "metrics": {
            "3way_accuracy": round(float(acc_3way), 4),
            "2way_accuracy": round(float(acc_2way), 4),
        },
        "paper_metrics": {
            "3way_accuracy": 0.725,
            "2way_accuracy": 0.895,
        },
        "gap_to_paper": {
            "3way": round(float(0.725 - acc_3way), 4),
            "2way": round(float(0.895 - acc_2way), 4),
        },
        "per_class": {
            label: {
                "gold_count": int(sum(1 for g in gold_labels if g == label)),
                "pred_count": int(sum(1 for p in pred_labels if p == label)),
                "correct":    int(sum(
                    1 for g, p in zip(gold_labels, pred_labels)
                    if g == label and p == label
                ))
            }
            for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        },
        "known_limitations": [
            "Index covers 79.8% of FEVER articles (23,733/29,756)",
            "Wikipedia 2023 dump vs paper's 2018 dump",
            "Val sample imbalance during training (48% NEI vs 40% true)",
            "Training with batch_size=1 — batched training not implemented"
        ],
        "training_summary": {
            "epochs":              3,
            "train_examples":      228277,
            "val_examples_per_epoch": 500,
            "epoch1_val_3way":     0.630,
            "epoch2_val_3way":     0.634,
            "epoch3_val_3way":     0.650,
        }
    }

    save_path = os.path.join(results_dir, "final_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {save_path}")
    return results


def print_qualitative_examples(gold_labels, pred_labels, test_data,
                                n_per_label=2):
    """
    Print n correct and n incorrect prediction examples per label.

    Args:
        gold_labels (list):  gold label strings
        pred_labels (list):  predicted label strings
        test_data (list):    original test examples
        n_per_label (int):   examples to show per label. Defaults to 2.
    """
    print("\nQUALITATIVE EXAMPLES")

    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        correct = [
            (g, p, test_data[i])
            for i, (g, p) in enumerate(zip(gold_labels, pred_labels))
            if g == label and p == g
        ][:n_per_label]

        incorrect = [
            (g, p, test_data[i])
            for i, (g, p) in enumerate(zip(gold_labels, pred_labels))
            if g == label and p != g
        ][:n_per_label]

        print(f"\n{'='*60}")
        print(f"LABEL: {label}")
        print(f"{'='*60}")

        print("  CORRECT predictions:")
        for g, p, ex in correct:
            print(f"    Claim: {ex['claim'][:80]}...")
            print(f"    Gold: {g} | Pred: {p}\n")

        print("  INCORRECT predictions:")
        for g, p, ex in incorrect:
            print(f"    Claim: {ex['claim'][:80]}...")
            print(f"    Gold: {g} | Pred: {p}\n")


if __name__ == "__main__":
    """
    Run full evaluation pipeline from command line:
        python evaluate.py
    """
    import yaml

    # paths
    paths = path_setup()
    apply_faiss_setup()

    with open(os.path.join(paths["config_dir"],
                           "fever_config.yaml")) as f:
        config = yaml.safe_load(f)

    MAX_LENGTH = config["model"]["max_source_length"]
    N_DOCS     = config["model"]["n_docs"]

    # load data
    print("Loading passages and index...")
    passages  = load_passages(paths["passages_path"])
    cpu_index = __import__("faiss").read_index(paths["faiss_index_path"])
    gpu_index = move_index_to_gpu(cpu_index)

    print("Loading FEVER test set...")
    dataset   = load_fever()
    test_data = list(dataset["test"])
    print(f"Test set: {len(test_data):,} examples")

    # show distribution
    dist = Counter(ex["label"] for ex in test_data)
    for label, count in dist.items():
        print(f"  {label}: {count:,} ({count/len(test_data)*100:.1f}%)")

    # load models
    print("\nLoading models from checkpoint...")
    q_encoder, q_tokenizer, model, bart_tokenizer = \
        load_models_from_checkpoint(paths["checkpoint_dir"])

    # run evaluation
    gold_labels, pred_labels, all_probs = run_evaluation(
        test_data, model, q_encoder,
        bart_tokenizer, q_tokenizer,
        gpu_index, passages,
        max_length=MAX_LENGTH, n_docs=N_DOCS
    )

    # metrics
    acc_3way, acc_2way, report = compute_metrics(
        gold_labels, pred_labels
    )

    print("\n" + "=" * 55)
    print("FINAL TEST SET RESULTS")
    print("=" * 55)
    print(f"3-way accuracy: {acc_3way:.1%}  (paper: 72.5%)")
    print(f"2-way accuracy: {acc_2way:.1%}  (paper: 89.5%)")
    print(f"\nTest examples:  {len(test_data):,}")
    print(f"2-way examples: "
          f"{sum(1 for g in gold_labels if g != 'NOT ENOUGH INFO'):,}"
          f" (excluding NEI)")
    print("\n" + "=" * 55)
    print("PER-CLASS REPORT")
    print("=" * 55)
    print(report)

    # plots and save
    plot_results(
        gold_labels, pred_labels, acc_3way, acc_2way,
        paths["results_dir"]
    )
    results = save_results(
        acc_3way, acc_2way,
        gold_labels, pred_labels,
        paths["results_dir"]
    )
    print_qualitative_examples(gold_labels, pred_labels, test_data)