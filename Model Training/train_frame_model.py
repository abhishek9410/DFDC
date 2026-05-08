import argparse
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


LABEL_TO_ID = {"REAL": 0, "FAKE": 1}
ID_TO_LABEL = {0: "REAL", 1: "FAKE"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_dataset_dir(path: Path) -> Path:
    if (path / "metadata.json").exists():
        return path

    sample_dir = path / "train_sample_videos"
    if (sample_dir / "metadata.json").exists():
        return sample_dir

    raise FileNotFoundError(
        f"No metadata.json found in {path} or {sample_dir}. "
        "Extract the DFDC sample zip first, or pass --dataset-dir to the video folder."
    )


def extract_zip_if_needed(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "train_sample_videos" / "metadata.json"
    if marker.exists():
        print(f"Dataset already extracted at {output_dir}")
        return
    print(f"Extracting {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def load_rows(dataset_dir: Path, max_videos: int | None, seed: int) -> list[dict]:
    dataset_dir = find_dataset_dir(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    rows = []
    for filename, item in metadata.items():
        path = dataset_dir / filename
        label = item.get("label")
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.exists() and label in LABEL_TO_ID:
            rows.append({"path": path, "filename": filename, "label": LABEL_TO_ID[label]})

    if not rows:
        raise ValueError(f"No labeled videos found in {dataset_dir}")

    random.Random(seed).shuffle(rows)
    if max_videos:
        rows = rows[:max_videos]
    return rows


def stratified_split(rows: list[dict], validation_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train_rows = []
    val_rows = []

    for label in sorted(LABEL_TO_ID.values()):
        label_rows = [row for row in rows if row["label"] == label]
        rng.shuffle(label_rows)
        val_count = max(1, round(len(label_rows) * validation_split)) if len(label_rows) > 1 else 0
        val_rows.extend(label_rows[:val_count])
        train_rows.extend(label_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    if not train_rows or not val_rows:
        raise ValueError("Not enough labeled videos to create train and validation splits.")
    return train_rows, val_rows


def clamp_box(x: int, y: int, w: int, h: int, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(width - x, max(0, int(w)))
    h = min(height - y, max(0, int(h)))
    return x, y, w, h


class VideoFrameSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        rows: list[dict],
        image_size: int,
        frames_per_video: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.rows = rows
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.samples = []
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self) -> None:
        self.samples = []
        for row_index, row in enumerate(self.rows):
            frame_count = self._frame_count(row["path"])
            if frame_count <= 0:
                continue
            indices = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=np.int32)
            for frame_index in indices.tolist():
                self.samples.append((row_index, int(frame_index)))
        if self.shuffle:
            self.rng.shuffle(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        images = np.zeros((len(batch), self.image_size, self.image_size, 3), dtype=np.float32)
        labels = np.zeros((len(batch),), dtype=np.int64)

        for item_index, (row_index, frame_index) in enumerate(batch):
            row = self.rows[row_index]
            images[item_index] = self._read_frame(row["path"], frame_index)
            labels[item_index] = row["label"]
        return images, labels

    def _frame_count(self, video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        return count

    def _read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        face = self._largest_face(frame)
        if face is not None:
            x, y, w, h = face
            margin = int(0.18 * max(w, h))
            x, y, w, h = clamp_box(x - margin, y - margin, w + (2 * margin), h + (2 * margin), frame.shape)
            if w > 0 and h > 0:
                frame = frame[y : y + h, x : x + w]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32)

    def _largest_face(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda box: box[2] * box[3])


def build_model(image_size: int, learning_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="frame")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.03)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)
    x = tf.keras.layers.RandomContrast(0.15)(x)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
        include_preprocessing=True,
    )
    backbone.trainable = False

    x = tf.keras.layers.Dropout(0.35)(backbone.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetv2_frame_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def write_metadata(output_model: Path, args: argparse.Namespace, train_rows: list[dict], val_rows: list[dict]) -> None:
    metadata = {
        "model_type": "efficientnetv2_frame_detector",
        "labels": ID_TO_LABEL,
        "fake_class_id": 1,
        "image_size": args.image_size,
        "frames_per_video": args.frames_per_video,
        "video_prediction": "mean frame probability",
        "train_videos": len(train_rows),
        "validation_videos": len(val_rows),
    }
    metadata_path = output_model.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved model metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Train a stronger EfficientNetV2 frame detector.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "Dataset" / "deepfake-detection-challenge",
        help="Folder containing train_sample_videos/metadata.json or metadata.json directly.",
    )
    parser.add_argument(
        "--extract-zip",
        type=Path,
        default=None,
        help="Optional DFDC sample zip to extract before training.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=root / "Deploy" / "models" / "efficientnetv2_frame_model.keras",
        help="Model path used automatically by Deploy/app.py when present.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.extract_zip:
        extract_zip_if_needed(args.extract_zip, args.dataset_dir)

    rows = load_rows(args.dataset_dir, args.max_videos, args.seed)
    train_rows, val_rows = stratified_split(rows, args.validation_split, args.seed)
    print(f"Training videos: {len(train_rows)}")
    print(f"Validation videos: {len(val_rows)}")

    train_data = VideoFrameSequence(
        train_rows,
        image_size=args.image_size,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_data = VideoFrameSequence(
        val_rows,
        image_size=args.image_size,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(args.image_size, args.learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        nested_backbone = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet")
            ),
            None,
        )
        tune_layers = nested_backbone.layers[-30:] if nested_backbone is not None else model.layers[-60:]
        for layer in tune_layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    model.save(args.output_model)
    write_metadata(args.output_model, args, train_rows, val_rows)
    print(f"Saved trained model to {args.output_model}")


if __name__ == "__main__":
    main()
