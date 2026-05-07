import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Model


LABEL_TO_ID = {"REAL": 0, "FAKE": 1}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_classifier_model(sequence_length: int = 20, feature_size: int = 2048) -> Model:
    input_features = Input(shape=(sequence_length, feature_size), name="input_3")
    input_mask = Input(shape=(sequence_length,), dtype="bool", name="input_4")

    x = GRU(16, return_sequences=True, name="gru")(input_features, mask=input_mask)
    x = GRU(8, return_sequences=False, name="gru_1")(x)
    x = Dropout(0.4, name="dropout")(x)
    x = Dense(8, activation="relu", name="dense")(x)
    outputs = Dense(2, activation="softmax", name="dense_1")(x)
    model = Model(inputs=[input_features, input_mask], outputs=outputs, name="functional_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_metadata(dataset_dir: Path, max_videos: int | None, seed: int) -> list[dict]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dataset_dir}")

    with metadata_path.open("r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    rows = []
    for filename, item in raw_metadata.items():
        video_path = dataset_dir / filename
        label = item.get("label")
        if label in LABEL_TO_ID and video_path.exists():
            rows.append({"filename": filename, "path": video_path, "label": LABEL_TO_ID[label]})

    if not rows:
        raise ValueError(f"No labeled videos found in {dataset_dir}")

    random.Random(seed).shuffle(rows)
    if max_videos:
        rows = rows[:max_videos]
    return rows


def extract_frames(video_path: Path, sequence_length: int, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    indices = np.linspace(0, frame_count - 1, sequence_length, dtype=np.int32)
    frames = []
    mask = []

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
            mask.append(False)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            margin = int(0.1 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            frame = frame[y : y + h, x : x + w]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame.astype(np.float32))
        mask.append(True)

    cap.release()
    return preprocess_input(np.asarray(frames, dtype=np.float32)), np.asarray(mask, dtype=bool)


def prepare_features(
    rows: list[dict],
    cache_path: Path,
    sequence_length: int,
    image_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["features"], cached["masks"], cached["labels"], cached["filenames"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    features = []
    masks = []
    labels = []
    filenames = []
    skipped = []

    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] extracting {row['filename']}")
        try:
            frames, mask = extract_frames(row["path"], sequence_length, image_size)
            video_features = extractor.predict(frames, batch_size=batch_size, verbose=0)
            features.append(video_features.astype(np.float32))
            masks.append(mask)
            labels.append(row["label"])
            filenames.append(row["filename"])
        except Exception as exc:
            skipped.append((row["filename"], str(exc)))
            print(f"Skipping {row['filename']}: {exc}")

    if not features:
        raise ValueError("Feature extraction failed for every video.")

    features_array = np.asarray(features, dtype=np.float32)
    masks_array = np.asarray(masks, dtype=bool)
    labels_array = np.asarray(labels, dtype=np.int64)
    filenames_array = np.asarray(filenames)

    np.savez_compressed(
        cache_path,
        features=features_array,
        masks=masks_array,
        labels=labels_array,
        filenames=filenames_array,
    )
    if skipped:
        skipped_path = cache_path.with_suffix(".skipped.json")
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)
    return features_array, masks_array, labels_array, filenames_array


def stratified_split(labels: np.ndarray, validation_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for label in sorted(set(labels.tolist())):
        indices = [i for i, y in enumerate(labels.tolist()) if y == label]
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * validation_split))) if len(indices) > 1 else 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def train_model(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    output_model: Path,
    epochs: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> None:
    train_idx, val_idx = stratified_split(labels, validation_split, seed)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Not enough videos to create train and validation splits.")

    model = build_classifier_model(sequence_length=features.shape[1], feature_size=features.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    model.fit(
        [features[train_idx], masks[train_idx]],
        labels[train_idx],
        validation_data=([features[val_idx], masks[val_idx]], labels[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    if output_model.exists():
        backup_path = output_model.with_suffix(".previous.h5")
        shutil.copy2(output_model, backup_path)
        print(f"Backed up previous model to {backup_path}")
    model.save(output_model)
    print(f"Saved trained model to {output_model}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Train the deployed deepfake classifier on Kaggle DFDC videos.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "Dataset" / "deepfake-detection-challenge" / "train_sample_videos",
        help="Directory containing Kaggle DFDC videos and metadata.json.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=root / "Dataset" / "processed" / "dfdc_inception_seq20.npz",
        help="Feature cache path. Delete this file to recompute features.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=root / "Deploy" / "models" / "inceptionNet_model.h5",
        help="Model path used by the Flask app.",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Limit videos for a quick smoke run.")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Extract/cache features without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    rows = load_metadata(args.dataset_dir, args.max_videos, args.seed)
    print(f"Loaded {len(rows)} videos from {args.dataset_dir}")
    features, masks, labels, _filenames = prepare_features(
        rows=rows,
        cache_path=args.cache_path,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(f"Feature cache shape: features={features.shape}, masks={masks.shape}, labels={labels.shape}")

    if args.prepare_only:
        return

    train_model(
        features=features,
        masks=masks,
        labels=labels,
        output_model=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
