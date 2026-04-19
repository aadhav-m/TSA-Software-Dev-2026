import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--model-out", default="my_model")
    parser.add_argument("--sequences", type=int, default=30)
    parser.add_argument("--frames",    type=int, default=10)
    parser.add_argument("--epochs",    type=int, default=150)
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--cv",        type=int, default=0)
    return parser.parse_args()


def load_dataset(data_dir, sign_labels, num_sequences, num_frames):
    all_sequences, all_labels = [], []
    for label_idx, sign_label in enumerate(sign_labels):
        for seq_idx in range(num_sequences):
            single_sequence = []
            for frame_idx in range(num_frames):
                frame_path = os.path.join(data_dir, sign_label, str(seq_idx), f"{frame_idx}.npy")
                if not Path(frame_path).exists():
                    single_sequence = []
                    break
                single_sequence.append(np.load(frame_path))
            if len(single_sequence) == num_frames:
                all_sequences.append(single_sequence)
                all_labels.append(label_idx)
    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def build_lstm_model(num_frames, num_keypoint_features, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh", input_shape=(num_frames, num_keypoint_features)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="tanh"),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def get_training_callbacks(model_out):
    return [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f"{model_out}_best.keras", monitor="val_categorical_accuracy", save_best_only=True, verbose=0),
    ]


def train_simple_split(X, Y_onehot, Y_raw, sign_labels, args):
    X_train, X_test, Y_train, Y_test, Y_train_raw, Y_test_raw = train_test_split(
        X, Y_onehot, Y_raw, test_size=args.test_size, random_state=42, stratify=Y_raw
    )
    model = build_lstm_model(args.frames, X.shape[2], len(sign_labels))
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              epochs=args.epochs, batch_size=args.batch, callbacks=get_training_callbacks(args.model_out))
    predictions = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(Y_test_raw, predictions, target_names=sign_labels))
    return model


def train_cross_validated(X, Y_onehot, Y_raw, sign_labels, args):
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    fold_accuracies = []
    best_model, best_accuracy = None, 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y_raw)):
        print(f"\nFold {fold_idx + 1}/{args.cv}")
        model = build_lstm_model(args.frames, X.shape[2], len(sign_labels))
        model.fit(X[train_idx], Y_onehot[train_idx], validation_data=(X[test_idx], Y_onehot[test_idx]),
                  epochs=args.epochs, batch_size=args.batch, callbacks=get_training_callbacks(args.model_out), verbose=0)
        predictions = np.argmax(model.predict(X[test_idx]), axis=1)
        fold_acc = np.mean(predictions == Y_raw[test_idx])
        fold_accuracies.append(fold_acc)
        if fold_acc > best_accuracy:
            best_accuracy = fold_acc
            best_model = model

    print(f"\nCV accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    return best_model


def main():
    args = parse_args()
    sign_labels = np.array(sorted(os.listdir(args.data_dir)))

    X, Y_raw = load_dataset(args.data_dir, sign_labels, args.sequences, args.frames)
    Y_onehot = to_categorical(Y_raw).astype(np.int32)

    trained_model = train_cross_validated(X, Y_onehot, Y_raw, sign_labels, args) if args.cv > 1 else train_simple_split(X, Y_onehot, Y_raw, sign_labels, args)

    trained_model.save(args.model_out)
    with open(f"{args.model_out}_labels.json", "w") as label_file:
        json.dump(sign_labels.tolist(), label_file)


if __name__ == "__main__":
    main()
