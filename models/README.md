# Historical CNN weights

`legacy_final_tuned_weights.h5` is copied from the original student submission. It contains **weights only**, not the entire compiled model. The matching architecture is in `emnist_cnn_experiments.ipynb` (Final Tuned CNN). It comes from the historical training run in which test data was reused for validation / EarlyStopping; **do not use its recorded 94.08% as independent test accuracy**.

The corrected `train_validation_safe.py` produces a separately trained model under `models/` after execution; `.gitignore` excludes that generated model by default.
