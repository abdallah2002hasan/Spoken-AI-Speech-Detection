from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_index
from features import extract


# In-memory cache. The same audio/feature vector is reused between SVM/RF and
# later robustness evaluation instead of being extracted again.
_FEATURE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def eer(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def metrics(y, pred, score):
    value, threshold = eer(y, score)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "eer": value,
        "eer_threshold": threshold,
    }


def grouped_split(df, test_size, seed):
    """Split by source-utterance group so one source cannot appear in both sets."""
    groups = df["group_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_i, test_i = next(splitter.split(df, df.label, groups))
    train_df = df.iloc[train_i].reset_index(drop=True)
    test_df = df.iloc[test_i].reset_index(drop=True)

    train_groups = set(train_df["group_id"].astype(str))
    test_groups = set(test_df["group_id"].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            "Data leakage detected: source-utterance groups overlap: "
            f"{sorted(overlap)[:10]}"
        )
    return train_df, test_df


def limit_source_groups(df, max_groups, seed):
    """Take a reproducible subset of complete source groups for quick/final runs."""
    if max_groups is None:
        return df.reset_index(drop=True)

    groups = np.array(sorted(df["group_id"].astype(str).unique()))
    if max_groups <= 0:
        raise ValueError("--max-groups must be greater than zero.")
    if max_groups >= len(groups):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected = set(rng.choice(groups, size=max_groups, replace=False).tolist())
    subset = df[df["group_id"].astype(str).isin(selected)].copy().reset_index(drop=True)
    return subset


def build_model(name, seed):
    if name == "svm":
        # probability=False is much faster. ROC-AUC/EER use decision_function().
        return (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            probability=False,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                # Compact but meaningful search for a course project.
                "clf__C": [1, 10],
                "clf__gamma": ["scale", 0.01],
                "clf__kernel": ["rbf"],
            },
        )

    # Scaling is unnecessary for Random Forest, so avoid that extra work.
    return (
        Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=seed,
                        # GridSearchCV handles outer parallelism; avoid nested overload.
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 20],
            "clf__min_samples_leaf": [1, 2],
        },
    )


def matrix(df, kind, cfg):
    vectors, kept = [], []
    total = len(df)
    progress_every = int(cfg.get("progress_every", 100))
    started = time.time()
    cache_hits = 0

    print(f"\n[{kind.upper()}] Feature extraction for {total:,} files...")

    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        path = str(row.path)
        key = (path, kind)
        try:
            if key in _FEATURE_CACHE:
                vec = _FEATURE_CACHE[key]
                cache_hits += 1
            else:
                vec = extract(path, kind, cfg)
                _FEATURE_CACHE[key] = vec

            vectors.append(vec)
            kept.append(pos - 1)
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")

        if pos == 1 or pos % progress_every == 0 or pos == total:
            elapsed = time.time() - started
            print(
                f"[{kind.upper()}] {pos:,}/{total:,} "
                f"({100.0 * pos / max(total, 1):5.1f}%) | "
                f"cache hits={cache_hits:,} | elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    if not vectors:
        raise ValueError(f"Feature extraction failed for every file in {kind} experiment.")

    return np.vstack(vectors), df.iloc[kept].reset_index(drop=True)


def tune_on_matrix(X_train, train_used, kind, model_name, cfg):
    """Select hyperparameters using ONLY the training partition."""
    y_train = train_used.label.values
    groups = train_used.group_id.astype(str)

    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two training groups are required for grouped CV.")

    requested_folds = int(cfg.get("cv_folds", 3))
    n_splits = min(requested_folds, unique_groups)
    if n_splits < 2:
        raise ValueError("Grouped CV requires at least two folds/groups.")

    cv = GroupKFold(n_splits=n_splits)
    estimator, grid = build_model(model_name, cfg["random_state"])

    n_candidates = 1
    for values in grid.values():
        n_candidates *= len(values)
    print(
        f"\nTraining {kind}+{model_name}: {n_candidates} parameter settings × "
        f"{n_splits} folds = {n_candidates * n_splits} fits",
        flush=True,
    )

    started = time.time()
    search = GridSearchCV(
        estimator,
        grid,
        scoring="f1",
        cv=cv,
        n_jobs=int(cfg.get("n_jobs", -1)),
        return_train_score=False,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    elapsed = time.time() - started

    print(
        f"Finished {kind}+{model_name}: CV F1={search.best_score_:.4f} "
        f"in {elapsed/60:.1f} min | params={search.best_params_}",
        flush=True,
    )

    return {
        "feature": kind,
        "classifier": model_name,
        "cv_f1": float(search.best_score_),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "n_train": len(y_train),
        "n_train_groups": int(unique_groups),
        "search": search,
    }


def tune_on_training(train_df, kind, model_name, cfg):
    X_train, train_used = matrix(train_df, kind, cfg)
    return tune_on_matrix(X_train, train_used, kind, model_name, cfg)


def model_scores(model, X):
    """Continuous scores for ROC-AUC/EER without expensive SVC probability fitting."""
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = np.asarray(score)
        if score.ndim > 1:
            score = score[:, 1]
        return score
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def evaluate_held_out(test_df, selection, cfg, tag):
    """Evaluate a pre-selected model exactly once on a held-out test partition."""
    kind = selection["feature"]
    model_name = selection["classifier"]
    model = selection["search"].best_estimator_

    X_test, test_used = matrix(test_df, kind, cfg)
    y_test = test_used.label.values
    pred = model.predict(X_test)
    score = model_scores(model, X_test)

    result = metrics(y_test, pred, score)
    result.update(
        {
            "experiment": tag,
            "feature": kind,
            "classifier": model_name,
            "selection_cv_f1": selection["cv_f1"],
            "n_train": selection["n_train"],
            "n_test": len(y_test),
            "n_train_groups": selection["n_train_groups"],
            "n_test_groups": int(test_used.group_id.astype(str).nunique()),
            "best_params": selection["best_params"],
        }
    )

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "AI"],
        yticklabels=["Real", "AI"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out / f"cm_{tag}_{kind}_{model_name}.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, score)
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(fpr, tpr, label=f"AUC={result['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"roc_{tag}_{kind}_{model_name}.png", dpi=180)
    plt.close()

    return result


def choose_best_candidate(train_df, cfg):
    """Compare all feature/classifier candidates using grouped CV on training only."""
    candidates = []
    for kind in ["mfcc", "lfcc", "spectral", "combined"]:
        # Extract this feature family ONCE, then reuse it for both classifiers.
        X_train, train_used = matrix(train_df, kind, cfg)
        for classifier in ["svm", "rf"]:
            selection = tune_on_matrix(X_train, train_used, kind, classifier, cfg)
            candidates.append(selection)
            print(
                f"CV candidate {kind}+{classifier}: "
                f"F1={selection['cv_f1']:.4f}",
                flush=True,
            )

    # Global model/feature selection is based only on training-set grouped CV.
    best = max(candidates, key=lambda row: row["cv_f1"])
    return candidates, best


def verify_disjoint(train_df, test_df, label):
    train_groups = set(train_df.group_id.astype(str))
    test_groups = set(test_df.group_id.astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"{label} leakage detected: overlapping groups {sorted(overlap)[:10]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset-root")
    ap.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help=(
            "Use only this many complete source-utterance groups. "
            "Useful for a quick test (e.g. 500) or practical final run (e.g. 2000)."
        ),
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override grouped CV folds (default from config, otherwise 3).",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root
    if args.cv_folds is not None:
        cfg["cv_folds"] = args.cv_folds

    cfg.setdefault("cv_folds", 3)
    cfg.setdefault("n_jobs", -1)
    cfg.setdefault("progress_every", 100)

    df = load_index(cfg["dataset_root"])
    full_groups = int(df.group_id.astype(str).nunique())
    full_rows = len(df)
    df = limit_source_groups(df, args.max_groups, cfg["random_state"])

    print("\n=== DATASET ===")
    print(df.groupby(["label", "generator"]).size())
    print(
        f"Rows used: {len(df):,}/{full_rows:,} | "
        f"source groups used: {df.group_id.astype(str).nunique():,}/{full_groups:,}"
    )
    if args.max_groups is not None:
        print(
            f"SUBSET MODE: reproducibly selected up to {args.max_groups:,} complete source groups."
        )

    # ------------------------------------------------------------------
    # STANDARD EXPERIMENT
    # 1) Create the held-out test split by source-utterance group.
    # 2) Select feature family, classifier, and hyperparameters using ONLY
    #    grouped cross-validation on the training partition.
    # 3) Evaluate the selected system exactly once on the held-out test set.
    # ------------------------------------------------------------------
    train, test = grouped_split(df, cfg["test_size"], cfg["random_state"])
    verify_disjoint(train, test, "Standard")

    print(
        f"\nStandard split: {train.group_id.nunique():,} train groups / "
        f"{test.group_id.nunique():,} test groups; overlap=0"
    )

    candidates, best = choose_best_candidate(train, cfg)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cv_rows = [
        {
            "feature": row["feature"],
            "classifier": row["classifier"],
            "cv_f1": row["cv_f1"],
            "n_train": row["n_train"],
            "n_train_groups": row["n_train_groups"],
            "best_params": row["best_params"],
        }
        for row in candidates
    ]
    cv_summary = pd.DataFrame(cv_rows).sort_values("cv_f1", ascending=False)
    cv_summary.to_csv(out / "model_selection_cv.csv", index=False)

    final_results = [evaluate_held_out(test, best, cfg, "standard_final")]

    # Save the standard final model selected without consulting test labels/results.
    joblib.dump(
        {
            "feature": best["feature"],
            "classifier": best["classifier"],
            "config": cfg,
            "model": best["search"].best_estimator_,
            "selection_cv_f1": best["cv_f1"],
            "best_params": json.loads(best["best_params"]),
        },
        out / "best_model.joblib",
    )

    # ------------------------------------------------------------------
    # UNSEEN-GENERATOR ROBUSTNESS EXPERIMENT
    # Standard winning feature/classifier family is fixed in advance.
    # Held-out generator is removed from training, and the unseen-generator
    # test remains source-group disjoint.
    # ------------------------------------------------------------------
    unseen = str(cfg.get("unseen_generator", "")).lower()
    if unseen and (df.generator.str.lower() == unseen).any():
        source_train, source_test = grouped_split(
            df, cfg["test_size"], cfg["random_state"]
        )
        gen_train = source_train[
            source_train.generator.str.lower() != unseen
        ].copy()
        gen_test = source_test[
            (source_test.label == 0)
            | (source_test.generator.str.lower() == unseen)
        ].copy()

        verify_disjoint(gen_train, gen_test, "Unseen-generator")

        if (gen_train.generator.str.lower() == unseen).any():
            raise RuntimeError(
                "Held-out generator unexpectedly appears in training data."
            )
        if gen_test.empty or not (gen_test.generator.str.lower() == unseen).any():
            raise ValueError(
                f"No '{unseen}' samples remained in the unseen-generator test split."
            )

        print(
            f"\nUnseen-generator experiment: hold out '{unseen}' from training; "
            f"group overlap=0"
        )

        unseen_selection = tune_on_training(
            gen_train,
            best["feature"],
            best["classifier"],
            cfg,
        )
        final_results.append(
            evaluate_held_out(
                gen_test,
                unseen_selection,
                cfg,
                f"unseen_{unseen}_final",
            )
        )

    summary = pd.DataFrame(final_results)
    summary.to_csv(out / "summary.csv", index=False)

    print("\n=== TRAINING-ONLY MODEL SELECTION ===")
    print(cv_summary.to_string(index=False))
    print("\n=== HELD-OUT FINAL RESULTS ===")
    print(summary.to_string(index=False))
    print(
        "\nSaved best model selected by grouped CV: "
        f"{best['feature']} + {best['classifier']} "
        f"(CV F1={best['cv_f1']:.4f})"
    )
    print(f"Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
