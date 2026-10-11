"""End-to-end pipeline: Network Anomaly & Intrusion Detection on NSL-KDD.

Usage:
    python main.py            # full run (~3-5 min on a laptop)
    python main.py --no-shap  # skip the SHAP interpretation step
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import evaluation as ev
from src.config import CLASS_ORDER, FIG_DIR, MODELS_DIR, RESULTS_DIR
from src.data import load_data
from src.interpretation import shap_analysis
from src.models import anomaly_detector, supervised_models
from src.preprocessing import preprocess


def main(run_shap: bool = True):
    for d in (RESULTS_DIR, FIG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. data
    print("[1/6] Loading NSL-KDD ...")
    train_df, test_df = load_data()
    print(f"      train: {train_df.shape}, test: {test_df.shape}")
    ev.plot_class_distribution(train_df, test_df, CLASS_ORDER, FIG_DIR / "class_distribution.png")

    # attack types that exist in test but never in train -> generalisation check
    novel_labels = sorted(set(test_df["label"]) - set(train_df["label"]))
    novel_mask = test_df["label"].isin(novel_labels).values
    known_attack_mask = (test_df["is_attack"].values == 1) & ~novel_mask

    # ------------------------------------------------------------ 2. preprocessing
    print("[2/6] Preprocessing (log-scaling, standardisation, one-hot) ...")
    X_train, X_test, feature_names, preprocessor = preprocess(train_df, test_df)
    print(f"      {X_train.shape[1]} features after encoding")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    y_train_bin, y_test_bin = train_df["is_attack"].values, test_df["is_attack"].values
    cls_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train_mc = train_df["attack_category"].map(cls_idx).values
    y_test_mc = test_df["attack_category"].map(cls_idx).values

    # ------------------------------------------------------------ 3. binary
    print("[3/6] Binary detection: normal vs attack ...")
    bin_rows, roc_scores, bin_preds, bin_models = {}, {}, {}, {}
    for name, model in supervised_models(2).items():
        t0 = time.time()
        model.fit(X_train, y_train_bin)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        bin_rows[name] = {**ev.binary_metrics(y_test_bin, pred, proba),
                          "train_time_s": time.time() - t0}
        roc_scores[name], bin_preds[name], bin_models[name] = proba, pred, model
        print(f"      {name:<20} F1={bin_rows[name]['f1']:.3f}  AUC={bin_rows[name]['roc_auc']:.3f}")

    # Unsupervised anomaly detection: learn "normal" only, flag deviations
    t0 = time.time()
    iso = anomaly_detector().fit(X_train[y_train_bin == 0])
    iso_score = -iso.score_samples(X_test)                # higher = more anomalous
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    bin_rows["Isolation Forest"] = {**ev.binary_metrics(y_test_bin, iso_pred, iso_score),
                                    "train_time_s": time.time() - t0}
    roc_scores["Isolation Forest"], bin_preds["Isolation Forest"] = iso_score, iso_pred
    print(f"      {'Isolation Forest':<20} F1={bin_rows['Isolation Forest']['f1']:.3f}  "
          f"AUC={bin_rows['Isolation Forest']['roc_auc']:.3f}  (unsupervised)")

    bin_df = pd.DataFrame(bin_rows).T.round(4)
    bin_df.to_csv(RESULTS_DIR / "metrics_binary.csv")
    ev.plot_metric_comparison(
        bin_df, ["accuracy", "precision", "recall (detection rate)", "f1", "roc_auc"],
        "Binary detection (normal vs attack) on KDDTest+", FIG_DIR / "binary_model_comparison.png")
    ev.plot_roc_curves(y_test_bin, roc_scores, FIG_DIR / "roc_curves.png")

    # Cross-validation on the TRAIN set only: shows how optimistic a random split is
    # compared to the official test set (which contains new attack types).
    cv = cross_val_score(supervised_models(2)["XGBoost"], X_train, y_train_bin,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="f1", n_jobs=1)
    cv_info = {"xgboost_5fold_cv_f1_mean": float(cv.mean()), "xgboost_5fold_cv_f1_std": float(cv.std()),
               "xgboost_test_f1": float(bin_df.loc["XGBoost", "f1"])}
    print(f"      XGBoost 5-fold CV F1 on train = {cv.mean():.4f} ± {cv.std():.4f} "
          f"vs F1 on KDDTest+ = {cv_info['xgboost_test_f1']:.4f}")

    best_bin = bin_df.drop(index="Isolation Forest")["f1"].idxmax()
    ev.plot_confusion(y_test_bin, bin_preds[best_bin], ["Normal", "Attack"],
                      f"Confusion matrix - {best_bin} (binary)",
                      FIG_DIR / "confusion_binary.png")
    joblib.dump(bin_models[best_bin], MODELS_DIR / "best_binary_model.joblib")

    seen_unseen = pd.DataFrame({
        name: {"Known attacks": p[known_attack_mask].mean(),
               "Novel attacks": p[novel_mask].mean()}
        for name, p in bin_preds.items()}).T
    seen_unseen.round(4).to_csv(RESULTS_DIR / "known_vs_novel_attacks.csv")
    ev.plot_seen_vs_unseen(seen_unseen, FIG_DIR / "known_vs_novel_attacks.png")

    # ------------------------------------------------------------ 4. multi-class
    print("[4/6] Multi-class classification: Normal / DoS / Probe / R2L / U2R ...")
    mc_rows, recalls, mc_preds = {}, {}, {}
    for name, model in supervised_models(len(CLASS_ORDER)).items():
        t0 = time.time()
        if name == "XGBoost":  # XGBoost has no class_weight -> balanced sample weights
            counts = np.bincount(y_train_mc)
            w = (len(y_train_mc) / (len(counts) * counts))[y_train_mc]
            model.fit(X_train, y_train_mc, sample_weight=w)
        else:
            model.fit(X_train, y_train_mc)
        pred = model.predict(X_test)
        mc_rows[name] = {**ev.multiclass_metrics(y_test_mc, pred),
                         "train_time_s": time.time() - t0}
        recalls[name] = recall_score(y_test_mc, pred, average=None,
                                     labels=range(len(CLASS_ORDER)), zero_division=0)
        mc_preds[name] = pred
        print(f"      {name:<20} macro-F1={mc_rows[name]['macro_f1']:.3f}")

    mc_df = pd.DataFrame(mc_rows).T.round(4)
    mc_df.to_csv(RESULTS_DIR / "metrics_multiclass.csv")
    recall_df = pd.DataFrame(recalls, index=CLASS_ORDER).T
    recall_df.round(4).to_csv(RESULTS_DIR / "per_class_recall.csv")
    ev.plot_metric_comparison(
        mc_df, ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
        "Multi-class classification on KDDTest+", FIG_DIR / "multiclass_model_comparison.png")
    ev.plot_per_class_recall(recall_df, FIG_DIR / "per_class_recall.png")

    best_mc = mc_df["macro_f1"].idxmax()
    ev.plot_confusion(y_test_mc, mc_preds[best_mc], CLASS_ORDER,
                      f"Confusion matrix - {best_mc} (multi-class)",
                      FIG_DIR / "confusion_multiclass.png")
    report = classification_report(y_test_mc, mc_preds[best_mc],
                                   target_names=CLASS_ORDER, digits=4, zero_division=0)
    (RESULTS_DIR / "classification_report_multiclass.txt").write_text(
        f"Best multi-class model: {best_mc}\n\n{report}")

    # ------------------------------------------------------------ 5. interpretation
    top_features = None
    if run_shap:
        print("[5/6] Interpreting XGBoost with SHAP ...")
        imp = shap_analysis(bin_models["XGBoost"], X_test, feature_names, FIG_DIR)
        imp.head(20).round(4).to_csv(RESULTS_DIR / "shap_top_features.csv", header=["mean_abs_shap"])
        top_features = imp.head(10).round(4).to_dict()
    else:
        print("[5/6] SHAP skipped")

    # ------------------------------------------------------------ 6. summary
    print("[6/6] Saving summary ...")
    summary = {
        "n_train": len(train_df), "n_test": len(test_df), "n_features": len(feature_names),
        "novel_attack_types_in_test": novel_labels,
        "cross_validation_vs_test": cv_info,
        "best_binary_model": best_bin, "best_multiclass_model": best_mc,
        "binary": bin_df.to_dict(orient="index"),
        "multiclass": mc_df.to_dict(orient="index"),
        "top_shap_features": top_features,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nBinary results:\n", bin_df.drop(columns="train_time_s"))
    print("\nMulti-class results:\n", mc_df.drop(columns="train_time_s"))
    print("\nPer-class recall:\n", recall_df.round(3))
    print(f"\nDone. Figures in {FIG_DIR}, metrics in {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-shap", action="store_true", help="skip SHAP interpretation")
    args = parser.parse_args()
    main(run_shap=not args.no_shap)
