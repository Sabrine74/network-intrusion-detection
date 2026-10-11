# Network Anomaly and Intrusion Detection with Machine Learning

Detecting malicious network connections on the **NSL-KDD** benchmark with supervised classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) and an unsupervised anomaly detector (Isolation Forest). The project covers the full workflow: **data preprocessing → model training → evaluation → interpretation of results (SHAP)**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Key findings

| | |
|---|---|
| **Best binary detector (supervised)** | XGBoost: F1 = 0.786, ROC-AUC = 0.971, false-alarm rate = 2.8 % |
| **Best detector of *unseen* attacks** | Isolation Forest (unsupervised): detects 67 % of novel attack types vs 39 % for XGBoost |
| **Optimism of random splits** | XGBoost reaches F1 = 0.999 in 5-fold CV on the training set but 0.786 on the official test set |
| **Hardest classes** | R2L and U2R (remote-to-local / user-to-root): rare in training, often look like normal traffic |
| **Most influential features (SHAP)** | `src_bytes`, `dst_bytes`, `dst_host_srv_count`, `count`, `dst_host_rerror_rate` |

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) (Canadian Institute for Cybersecurity) is a cleaned version of KDD Cup 99: duplicate records are removed, and the official test set (`KDDTest+`) contains **17 attack types that never appear in training**, which makes it a realistic test of generalisation.

- 125,973 training connections, 22,544 test connections
- 41 features per connection: basic TCP features (`duration`, `protocol_type`, `service`, `flag`, `src_bytes`…), content features (`hot`, `num_failed_logins`, `root_shell`…) and traffic-window features (`count`, `srv_count`, `dst_host_*` rates)
- Labels grouped into 5 classes:

| Class | Meaning | Examples |
|---|---|---|
| Normal | legitimate traffic | — |
| DoS | Denial of Service | neptune, smurf, teardrop |
| Probe | scanning / reconnaissance | portsweep, nmap, satan |
| R2L | Remote-to-Local: unauthorised access from a remote machine | guess_passwd, warezmaster |
| U2R | User-to-Root: privilege escalation | buffer_overflow, rootkit |

The data is downloaded automatically on first run.

![Class distribution](results/figures/class_distribution.png)

## Method

### 1. Preprocessing (`src/preprocessing.py`)
- Drop `num_outbound_cmds` (constant, always 0) and the `difficulty` column (not a traffic feature).
- `log1p` transform on heavy-tailed counters (`duration`, `src_bytes`, `dst_bytes`), then standardisation of all numeric features.
- One-hot encoding of `protocol_type`, `service`, `flag` (unknown categories in test → all-zeros) → 121 features.
- The transformer is **fitted on the training set only** and then applied to the test set (no data leakage).

### 2. Models (`src/models.py`)
- **Supervised:** Logistic Regression, Decision Tree, Random Forest, XGBoost. Class imbalance is handled with `class_weight="balanced"` (sample weights for XGBoost in the multi-class task).
- **Unsupervised anomaly detection:** Isolation Forest trained **only on normal traffic**; any connection that deviates from it is flagged as an attack.

### 3. Tasks
- **Binary detection:** normal vs attack.
- **Multi-class classification:** Normal / DoS / Probe / R2L / U2R.

### 4. Evaluation (`src/evaluation.py`)
Accuracy, precision, recall (detection rate), F1, false-alarm rate, ROC-AUC; macro-F1 and per-class recall for the multi-class task; confusion matrices; detection rate on known vs novel attack types; 5-fold cross-validation on the training set.

### 5. Interpretation (`src/interpretation.py`)
SHAP (TreeExplainer) on the XGBoost binary model to explain which features drive the "attack" decision and in which direction.

## Results

### Binary detection (KDDTest+)

| Model | Accuracy | Precision | Recall | F1 | False-alarm rate | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.753 | 0.914 | 0.624 | 0.742 | 7.7 % | 0.781 |
| Decision Tree | 0.786 | 0.956 | 0.654 | 0.777 | 3.9 % | 0.807 |
| Random Forest | 0.777 | 0.968 | 0.629 | 0.762 | 2.7 % | 0.961 |
| **XGBoost** | **0.795** | **0.969** | 0.661 | 0.786 | 2.8 % | **0.971** |
| Isolation Forest *(unsupervised)* | 0.797 | 0.940 | **0.687** | **0.794** | 5.8 % | 0.946 |

![Binary comparison](results/figures/binary_model_comparison.png)

<p float="left">
  <img src="results/figures/roc_curves.png" width="48%" />
  <img src="results/figures/confusion_binary.png" width="48%" />
</p>

### Generalisation to unseen attacks

![Known vs novel attacks](results/figures/known_vs_novel_attacks.png)

Supervised models learn the *signatures* of the attacks they were trained on: their detection rate drops from ~75 % on known attack types to 28–43 % on the 17 new ones (e.g. `mscan`, `apache2`, `snmpguess`). Isolation Forest only models *normal* behaviour, so its detection rate is almost the same on known (69 %) and novel (67 %) attacks.

### Multi-class classification (KDDTest+)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| **Logistic Regression** | 0.781 | 0.658 | **0.595** | **0.596** |
| Decision Tree | 0.748 | 0.644 | 0.496 | 0.475 |
| Random Forest | 0.735 | 0.769 | 0.469 | 0.470 |
| XGBoost | **0.784** | **0.835** | 0.535 | 0.552 |

![Per-class recall](results/figures/per_class_recall.png)
![Multi-class confusion matrix](results/figures/confusion_multiclass.png)

### Interpretation (SHAP)

<p float="left">
  <img src="results/figures/shap_importance.png" width="48%" />
  <img src="results/figures/shap_beeswarm.png" width="48%" />
</p>

- **`src_bytes` / `dst_bytes`** dominate: very small payloads (blue dots on the right) push towards *attack*, typical of SYN floods and scans where connections carry almost no data.
- **`count`** (connections to the same host in the last 2 s) and **`dst_host_rerror_rate`** (share of rejected connections) are high during DoS and port scans.
- **`dst_host_srv_count`** high → normal: many past connections to the same service from a host indicate routine traffic.
- **`hot`**, **`service_private`**, **`protocol_type_icmp`**, **`service_ecr_i`** (ICMP echo) are strong attack indicators when present (ICMP floods like `smurf`, `pod`).

## Discussion

1. **Random splits are misleading.** XGBoost reaches F1 = 0.999 in 5-fold CV on the training data but 0.786 on KDDTest+. The gap comes from the 17 novel attack types and the different class mix in the test set — a model must be evaluated on traffic it has not seen.
2. **High precision, moderate recall.** Supervised models raise few false alarms (≈3 %) but miss about a third of attacks, mostly novel ones and R2L/U2R.
3. **R2L and U2R are the hard classes.** They represent <1 % of training data, and their connections look like normal traffic at the network level (a password-guessing session or a privilege escalation runs over normal-looking TCP sessions). More than half of R2L/U2R test connections are predicted as Normal.
4. **Complementary approaches.** Supervised models are better at recognising *known* attacks with few false alarms; anomaly detection is better at catching *new* ones. A practical IDS would combine both (e.g. signature-like classifier + anomaly score).
5. **Accuracy is not enough.** With imbalanced classes, macro-F1 and per-class recall reveal weaknesses that overall accuracy hides (XGBoost has the best accuracy but only 7 % U2R recall).

### Possible improvements
- Oversampling of rare classes (SMOTE) or cost-sensitive learning for R2L/U2R
- Hyperparameter search (Optuna / GridSearchCV) and threshold tuning to trade false alarms for detection rate
- Hybrid model: supervised classifier + Isolation Forest / autoencoder score
- Evaluation on a more recent dataset (CIC-IDS2017, UNSW-NB15)

## Project structure

```
network-intrusion-detection/
├── main.py                  # end-to-end pipeline
├── src/
│   ├── config.py            # paths, column names, attack → category mapping
│   ├── data.py              # download + loading
│   ├── preprocessing.py     # log-scaling, standardisation, one-hot encoding
│   ├── models.py            # classifiers + Isolation Forest
│   ├── evaluation.py        # metrics and figures
│   └── interpretation.py    # SHAP analysis
├── results/
│   ├── figures/             # all plots shown above
│   ├── metrics_binary.csv
│   ├── metrics_multiclass.csv
│   ├── per_class_recall.csv
│   ├── known_vs_novel_attacks.csv
│   ├── shap_top_features.csv
│   ├── classification_report_multiclass.txt
│   └── summary.json
├── data/                    # NSL-KDD (downloaded automatically, not versioned)
├── models/                  # saved models (not versioned)
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/<your-username>/network-intrusion-detection.git
cd network-intrusion-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                   # ~2-3 min; add --no-shap to skip SHAP
```

All metrics and figures are regenerated in `results/`. Results are reproducible (`random_state = 42`).

## Author

**Sabrine Assili** — Telecommunications engineering student (ENET'Com Sfax, Automotive Connectivity Engineering)

## License

MIT — see [LICENSE](LICENSE).
