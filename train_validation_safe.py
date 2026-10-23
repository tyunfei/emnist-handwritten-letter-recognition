"""Train an EMNIST Letters CNN with separate train, validation, and test splits.

This is a corrected, not-yet-executed companion to the historical notebook.
It requires the coursework-compatible EMNIST CSV; scores will differ on rerun.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

SEED = 42


def load_csv(path: Path):
    """Read course-format CSV: column 0 is class 1..26, followed by 784 pixels."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}. See README.md")
    df = pd.read_csv(path, header=None, dtype=np.float32)
    if df.shape[1] != 785:
        raise ValueError(f"Expected 785 columns (label + 784 pixels), found {df.shape[1]}")
    valid = df.iloc[:, 0].between(1, 26)
    removed = int((~valid).sum())
    df = df.loc[valid]
    if df.empty:
        raise ValueError("No valid class labels 1..26 found")
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True).reshape(-1, 28, 28)
    # Orientation correction matches the final preprocessing in the original notebook.
    X = np.rot90(X, k=3, axes=(1, 2)).copy()
    X = (X / 255.0)[..., np.newaxis]
    y = df.iloc[:, 0].to_numpy(dtype=np.int32) - 1
    print(f"Loaded {len(df):,} valid samples, excluded {removed:,} invalid-label rows")
    return X, y


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', metrics=['accuracy'],
    )
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('emnist-letters.csv'))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y = load_csv(args.data)
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.20,
        random_state=SEED, stratify=y_train_all,
    )
    print(f"Train: {len(X_train):,}  Validation: {len(X_val):,}  Test: {len(X_test):,}")
    y_train = to_categorical(y_train, num_classes=26)
    y_val = to_categorical(y_val, num_classes=26)
    y_test = to_categorical(y_test, num_classes=26)

    augmentation = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1,
    )
    model = build_model()
    model.fit(
        augmentation.flow(X_train, y_train, batch_size=args.batch_size, seed=SEED),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=1,
    )
    # Do not use the test set as validation or for EarlyStopping.
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Held-out evaluation from this run — accuracy: {accuracy:.4%}, loss: {loss:.4f}")
    out = Path('models')
    out.mkdir(exist_ok=True)
    model.save(str(out / 'validation_safe_cnn.h5'))
    print('Saved models/validation_safe_cnn.h5 (excluded from Git by .gitignore)')


if __name__ == '__main__':
    main()
