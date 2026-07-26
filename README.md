# Credit Card Fraud Detection — IEEE-CIS

End-to-end fraud detection on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset (~590K transactions, 27:1 class imbalance). The project builds and compares three model families on a **leakage-safe, time-aware evaluation setup**, then translates model scores into operational decisions under realistic review-capacity constraints.

**Headline result:** XGBoost achieves **PR-AUC 0.507** on a held-out *future* time period (vs. a ~0.035 no-skill baseline). Under a realistic 1%-of-transactions review budget, it catches **25.6% of all fraud with 89% precision** — i.e. reviewing only 1 in 100 transactions surfaces fraud correctly 9 times out of 10.

---

## Why this project

Most fraud-detection portfolio projects train a classifier on a random split and report a high accuracy or ROC-AUC. Two things make those results misleading, and this project is built around avoiding both:

1. **Temporal leakage.** Transaction data is time-ordered; a random split lets the model train on the future and predict the past, inflating scores that then collapse in production. This project splits strictly by time and verifies it.
2. **Accuracy under imbalance.** At 3.5% fraud, a model predicting "never fraud" scores 96.5% accuracy and is useless. This project evaluates on PR-AUC and frames the final decision as a cost/capacity trade-off, not a threshold default.

---

## Dataset

The IEEE-CIS dataset contains 590,540 transactions with 3.5% labelled fraud (≈27:1 imbalance), spanning ~182 days. Features include transaction amount, product/card/address attributes, counting features (`C1–C14`), time-delta features (`D1–D15`), and 339 anonymized Vesta-engineered features (`V1–V339`), plus an identity/device block available for ~24% of transactions.

Because the competition's test labels are not public, all three of my splits (train/validation/test) are carved from the labelled training data by time.

**Reproduce data download:**
```bash
kaggle competitions download -c ieee-fraud-detection -p data/
unzip data/ieee-fraud-detection.zip -d data/
```

---

## Key EDA findings

| Finding | Detail |
|---|---|
| **Severe imbalance** | 3.5% fraud (~27:1), stable across the full time span |
| **Missingness is informative** | Transactions *with* identity data are fraudulent 7.96% of the time vs. 2.11% without — a ~4x gap, so missingness was treated as signal, not noise |
| **Categorical signal** | Product "C" (11.7% fraud) and credit cards (6.7%) are high-*rate* segments; product "W" (75% of volume) drives the most fraud in absolute *count* — rate ≠ volume |
| **Amount is weak alone** | Fraud/legit amount distributions overlap heavily; useful only in combination |
| **Strong time structure** | Fraud rate swings ~4.5x by hour — driven by legitimate traffic collapsing overnight while fraud stays roughly constant |

<!-- Suggested images to add:
     - fraud_rate_by_hour.png  (dual-axis: fraud rate vs transaction volume)
     - shap_summary.png        (beeswarm) -->

The hour finding is worth highlighting: fraud rate peaks not because fraud surges overnight, but because honest users are asleep, so the same volume of fraud becomes a much larger *share*.

---

## Methodology

### Time-aware split (the core of the project)
Data sorted by `TransactionDT`, then cut chronologically: earliest 70% → train, next 15% → validation, final 15% → test. Each block's time range is verified to sit strictly after the previous one. The test set is sealed until final evaluation and informs **no** modeling decision.

Fraud rate is stable across splits (3.52% / 3.43% / 3.48%), so score differences reflect model quality, not shifting base rates.

### Leakage-safe feature pipeline
A `fit`/`transform` pipeline (in `src/features.py`) mirrors scikit-learn's design: all learned statistics (e.g. per-card amount means/stds) are **fitted on train only** and applied forward to validation/test. This makes leakage structurally impossible — the pipeline regenerates the full feature set from raw data in one call.

Engineered features include:
- One-hot encoding of low-cardinality categoricals (`ProductCD`, `card4`, `card6`)
- Missing-indicators for columns where missingness was shown to separate fraud
- **Card-level amount z-score** — "is this amount unusual *for this card*" — turning weak raw amount into a contextual signal

### Handling imbalance
`scale_pos_weight` (XGBoost) / `class_weight` (LR) / weighted `BCEWithLogitsLoss` (MLP), all ≈27:1.

---

## Models & results

Three model families trained on the identical leakage-safe feature set, evaluated on the **sealed test set**:

| Model | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.827 | 0.175 |
| Neural Net (MLP) | 0.833 | 0.249 |
| **XGBoost** | **0.891** | **0.507** |

**XGBoost wins decisively** (~2x the MLP, ~2.9x LR on PR-AUC) — the expected result for tabular data, where gradient-boosted trees consistently outperform neural networks.

**XGBoost is also the most robust to temporal drift.** Between validation and the future test period, XGBoost lost only 6.5% of its PR-AUC (0.542 → 0.507), while the MLP lost ~45% (0.449 → 0.249). Trees partition locally, so drift in one region doesn't corrupt the whole function — a meaningful property for a model that must survive between retrains.

---

## From scores to decisions

A fraud model outputs probabilities, but a real system must make binary flag/no-flag calls. The default 0.5 threshold implicitly assumes false positives and false negatives cost the same — false in fraud.

**Cost-based threshold:** modelling a missed fraud as costing the transaction amount and a false alarm as a fixed review cost yields a cost-minimizing threshold of **0.30** (83% recall). But **sensitivity analysis** shows the optimum swings from 0.10 to 0.85 depending on the assumed false-alarm cost — so a single threshold isn't defensible without a business-supplied cost.

**Capacity-constrained reframing (the operational result):** the cost-optimal point would require reviewing 20% of all transactions — infeasible. Framing detection as "catch the most fraud subject to a review budget" is how fraud teams actually operate:

| Review budget | Recall | Precision |
|---|---|---|
| 0.5% of transactions | 13.4% | 93.4% |
| **1.0% of transactions** | **25.6%** | **89.2%** |
| 2.0% of transactions | 39.0% | 67.9% |
| 5.0% of transactions | 54.0% | 37.6% |

At the top of the ranking the model is highly precise — its most confident predictions are almost all genuine fraud, exactly what a triage system needs.

---

## Interpretation (SHAP)

SHAP analysis on the trained XGBoost:

- **Vesta's counting features (`C13`, `C14`, `C1`, `C5`) are the backbone** — the model's most-used signals.
- **The engineered card-level amount features rank 7th and 10th of 426** — above most raw features — validating the EDA-driven feature work. The card-type encoding (`card6_debit`, rank 13) independently confirms the debit/credit risk difference found in EDA.
- **`TransactionAmt` ranks 4th** despite weak *marginal* separation — a concrete example of a feature whose predictive value emerges only in combination with others.
- **The explicit missing-indicators were redundant (near-zero SHAP).** XGBoost handles missingness natively by learning split directions for NaNs, so it already captured the missingness signal through the raw columns. These indicators would benefit only the imputation-based models (LR, MLP), which lose that signal — a model-family-dependent result.

---

## Limitations & future work

- **String categoricals dropped.** 28 high-cardinality string columns (email domains, device info, `M`-flags) were dropped for the first pass; several likely carry signal and could be target/frequency-encoded.
- **Untuned models.** Sensible defaults were used; hyperparameter search (e.g. Optuna) would likely add marginal PR-AUC.
- **Linear cost model.** The threshold analysis assumes a fixed per-alarm cost; a capacity-bounded formulation better reflects reality.
- **Deployment.** A Streamlit/FastAPI demo would make the model interactive.

**Next:** this project is the foundation for a follow-up on **adversarial robustness** — stress-testing these trained models against evasion attacks (perturbed transactions designed to slip past detection), based on the security-evaluation framework in Xiao et al. (2023), *INFORMS Journal on Computing*.

---

## Repo structure

```
├── data/                     # raw CSVs + parquet (gitignored)
├── notebooks/
│   └── fraud_detection.ipynb # full analysis, EDA → modelling → SHAP
├── src/
│   └── features.py           # leakage-safe fit/transform feature pipeline
├── models/                   # trained models (xgb, mlp) + preprocessing
└── README.md
```

## Reproduce

```bash
python -m venv venv && source venv/Scripts/activate   # (Windows Git Bash)
pip install -r requirements.txt
# download data (see Dataset section), then run the notebook top to bottom
```

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · PyTorch · SHAP · matplotlib

---

*Built by Lakshya Kaviya — github.com/lakshyakaviya · lakshyakaviya2003@gmail.com*
