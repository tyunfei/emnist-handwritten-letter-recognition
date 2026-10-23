# Evaluation and publication notes

## What the uploaded final notebook actually recorded

The relevant original cells report:

| Experiment | Accuracy | Loss | Source (original notebook index, zero-based) |
|---|---:|---:|---|
| Baseline CNN | 92.30% | 0.2390 | 56 |
| Baseline + augmentation | 92.97% | 0.2080 | 86 |
| Improved CNN | 93.09% | 0.2224 | 105 |
| Improved CNN + augmentation | 93.54% | 0.1948 | 108 |
| Tuned CNN | 94.08% | 0.1792 | 162 |
| Tuned model reloaded | 94.08% | 0.1792 | 166 |

### What was wrong and what this portfolio edition changed

1. Original cell 113 quoted **93.69%** for improved+augmentation, although evaluation cell 108 showed **93.54%**. The portfolio notebook corrects this statement.
2. Original cell 163 incorrectly said that the tuned CNN *did not* beat the earlier improved+augmentation model and assigned 94.08% to that earlier model. That text was corrected.
3. The summary table in original cell 164 mismatched some model names and results. The portfolio version uses the recorded per-model evaluation outputs.
4. Original cell 160 trains the tuned CNN with **`validation_data=(X_test, y_test)`**. The evaluation in cell 162 and saved-model reload in cell 166 use **the same** `X_test`. Thus 94.08% cannot be described as independent/untouched test accuracy. The original code is preserved, but the portfolio notebook adds notices and a new corrected training script.
5. In the original notebooks, multiple alternatives were evaluated using the same test set; selection based on those scores can bias an estimate of final generalisation.

This package keeps the model training and saved outputs as **historical artifacts**, not newly reproduced validation results. The model weights under `models/` correspond to that historical tuned run and inherit its limitations.

### Corrected pipeline

`train_validation_safe.py` is a separate training example for future reruns. It reserves an independent test split, then makes validation from only training data. Its reported score is **unknown** until executed. To compare against external results rigorously after having consulted historical test outcomes, use new truly independent examples.

### Publication checklist

- [ ] Confirm your school permits publishing your assessment-derived source code and outputs.
- [ ] Confirm your dataset, trained weights and any third-party content may be redistributed as applicable; the original coursework CSV is intentionally **not** included.
- [ ] Check plots/notebook output for any additional identifying or sensitive content.
- [ ] Run the corrected script locally with a dataset you are authorised to use.
- [ ] Avoid quoting the historical 94.08% as a held-out result.
- [ ] Once published, link this repository from your GitHub profile README.

The original submission archive was not modified, and no content was uploaded to GitHub by this preparation step.