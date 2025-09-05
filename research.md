Traditional Dance Video Classification — Codebase Overview

- Purpose: Build a video classification system for traditional dance gestures and report the results in the accompanying paper (PDF).
- Primary methods implemented: I3D-Inception (3D CNN) and VGG16 + LSTM (2D CNN features + temporal modeling).
- Data pipeline: Video frame sequence generation via custom Keras-compatible generators.
- Outputs: Trained models, metrics (accuracy, precision, recall, F1), and confusion matrices for the dataset.

Project structure
- Notebooks: End-to-end experiments and analysis
  - `notebooks/tari_i3d.ipynb`: I3D pipeline, training, evaluation, metrics.
  - `notebooks/tari_vgg16_lstm.ipynb`: VGG16+LSTM pipeline at 128×128 resolution.
  - `notebooks/tari_vgg16_lstm_224.ipynb`: VGG16+LSTM pipeline at 224×224 resolution.
  - `i3d.ipynb`, `vgg16_lstm.ipynb`: Generic versions of the above pipelines.
- Model implementation:
  - `notebooks/i3d_inception.py` and `lib/i3d_inception/main.py`: I3D Inception v1 (Inflated 3D) Keras model with optional pretrained weights.
- Data utilities:
  - `lib/keras_video/generator.py`: `VideoFrameGenerator` for sampling N frames per video and producing Keras `Sequence` batches.
  - `lib/keras_video/flow.py`: `OpticalFlowGenerator` for motion representations (Farneback optical flow and variants).
  - `lib/helpers.py`: Helpers for class discovery, glob pattern generation, prediction aggregation, and score calculation.
- Checkpoints and history: Saved models and training artifacts.

Dataset and class setup
- Folder structure (typical):
  - `DATASET_DIR/train/<ClassName>/*.mp4`
  - `DATASET_DIR/val/<ClassName>/*.mp4`
  - `DATASET_DIR/test/<ClassName>/*.mp4`
- Class discovery and glob patterns use helpers:
  - `helpers.get_generated_class_names(dataset_dir, data_type)`
  - `helpers.get_generated_glob_pattern(dataset_dir, data_type)`
- Alternative single-root pattern (without train/val/test subfolders) is supported in some notebooks using a split on-the-fly.

What videos are classified
- Focus: Short video clips of fundamental Balinese traditional dance movements (Dasar Gerakan Tari Bali). Class labels are taken directly from folder names in the dataset(s), so the exact set may vary by source.
- Typical dataset sources in the notebooks:
  - `Dasar-Gerakan-Tari-Bali-All-Men`
  - `Dasar-Gerakan-Tari-Bali-All-Women`
- Common class label sets observed in this repository:
  - 7-class subset:
    - `Agem_Kanan`, `Agem_Kiri`, `Ngegol`, `Nyalud`, `Nyeregseg`, `Seledet`, `Ulap_Ulap`
  - 11-class set (example from All-Men dataset):
    - `Agem_Kanan`, `Agem_Kiri`, `Gandang_Gandang`, `Malpal`, `Nayog`, `Nepuk_Kampuh`, `Oyod`, `Piles`, `Seledet`, `Tapak_Sirang_Pada`, `Ulap_Ulap`
  - 13-class set (example from All-Women dataset):
    - `Ngelung`, `Nyeregseg`, `Seledet`, `Tapak_Sirangpada`, `Ngeseh`, `Ngegol`, `Agem_Kanan`, `Ngelo`, `Ngumbang`, `Nyalud`, `Ulap_Ulap`, `Agem_Kiri`, `Ngeed`
- Notes:
  - Underscores reflect directory naming; they correspond to specific named movements (e.g., Agem, Seledet, Tapak Sirang Pada).
  - If you provide a dataset with different class folders, the system will automatically learn and predict those classes.

Data pipeline
- Core generator: `lib/keras_video/generator.VideoFrameGenerator`
  - Inputs: class list, `glob_pattern` with `{classname}`, number of frames `nb_frames` (e.g., 30), `target_shape` (e.g., 224×224), `nb_channel` (RGB=3), `batch_size`.
  - Splitting: `split_val` and `split_test` to automatically partition files (e.g., 20% val, 20% test) while keeping the rest for training.
  - Sampling strategy: Evenly samples `nb_frames` across each video using header-estimated total frame count; falls back to a safe full-read if header is unreliable.
  - Returns: `numpy` arrays shaped `(batch, nb_frames, H, W, C)` for images and one-hot labels with `classes_count` columns.
- Optional motion representation: `lib/keras_video/flow.OpticalFlowGenerator` builds optical-flow-like inputs using Farneback or difference masks.

Methods
1) I3D Inception (Inflated 3D ConvNet)
- Source: `notebooks/i3d_inception.py`, `lib/i3d_inception/main.py` exposing `Inception_Inflated3d`.
- Backbone: Inception-v1 inflated to 3D convolutions for spatiotemporal modeling.
- Pretrained weights: Kinetics RGB/Flow or ImageNet+Kinetics variants (downloaded automatically via Keras `get_file`).
- Typical usage (feature extractor):
  - `Inception_Inflated3d(include_top=False, weights='rgb_kinetics_only', input_shape=(T, 224, 224, 3), classes=<num>)`
  - Freeze backbone (`model.trainable = False`), then add classification head:
    - `GlobalAveragePooling3D → Dense(512, relu) → Dropout(0.5) → Dense(num_classes, softmax)`
- Training configuration (from notebooks):
  - Loss: `categorical_crossentropy`
  - Optimizer: `adam`
  - Metrics: `acc`
  - Callbacks: `EarlyStopping(monitor='loss', patience=10)` and `ModelCheckpoint('Checkpoint/i3d.h5', monitor='val_acc', save_best_only=True, mode='max')`
  - Fit: `model.fit(train, validation_data=valid, epochs=100, callbacks=callbacks)`

2) VGG16 + LSTM
- Visual backbone: `VGG16(include_top=False, weights='imagenet')` applied per frame via `TimeDistributed`.
- Temporal head: `TimeDistributed(GlobalAveragePooling2D) → LSTM(256) → Dense(1024, relu) → Dropout(0.2) → Dense(num_classes, softmax)`
- Input shape: `(NBFRAME, H, W, 3)` with typical NBFRAME=30; H×W either 128×128 or 224×224 depending on notebook.
- Training configuration (from notebooks):
  - Loss: `categorical_crossentropy`
  - Optimizer: `adam`
  - Metrics: `acc`
  - Callbacks (examples):
    - `EarlyStopping(monitor='loss', patience=10)`
    - `ModelCheckpoint('Checkpoint/vgg16-lstm-224.h5', monitor='val_acc', save_best_only=True, mode='max')`
  - Fit: `model.fit(train, validation_data=valid, epochs=100, callbacks=callbacks)`

Evaluation
- Inference and label aggregation:
  - `helpers.get_populated_y_data(generator=test, batch_size=BS, model=model)`
  - Converts batched softmax predictions to `argmax` labels and aligns with test labels.
- Metrics:
  - Accuracy, precision (macro), recall (macro), F1 (macro) via `helpers.get_calculated_score` or directly using `sklearn.metrics`.
- Confusion matrix and classification report:
  - `confusion_matrix(y_true, y_pred)` and `classification_report(..., target_names=test.classes)` visualized with Seaborn heatmaps.

Experimental results (from notebooks)
- I3D (tari_i3d.ipynb):
  - Example metrics: Accuracy ≈ 0.788–0.803, Precision ≈ 0.811–0.815, Recall ≈ 0.785–0.802, F1 ≈ 0.783–0.804.
- VGG16+LSTM (tari_vgg16_lstm.ipynb, 128×128):
  - Example metrics: Accuracy ≈ 0.742, Precision ≈ 0.766, Recall ≈ 0.742, F1 ≈ 0.745.
- VGG16+LSTM (tari_vgg16_lstm_224.ipynb, 224×224):
  - Example metrics: Accuracy ≈ 0.826, Precision ≈ 0.833, Recall ≈ 0.827, F1 ≈ 0.823.
- Notes:
  - Metrics vary with dataset splits, class sets, and preprocessing. Use the confusion matrix and reports in the notebooks for per-class insight.

High-level training flow (Mermaid)
```mermaid
flowchart TD
  A[Raw videos organized by class] --> B[VideoFrameGenerator\n- split_val, split_test\n- nb_frames, target_shape]
  B --> C{Method}
  C -->|I3D| D[I3D Inception (include_top=False)\nweights: Kinetics]
  C -->|VGG16+LSTM| E[TimeDistributed(VGG16)\nGlobalAvgPool2D→LSTM→Dense]
  D --> F[Classifier Head\nGAP3D→Dense(512)→Dropout→Dense(num_classes, softmax)]
  E --> G[Softmax Head]
  F --> H[Compile\nloss=categorical_crossentropy\noptimizer=adam\nmetrics=acc]
  G --> H
  H --> I[Fit/train\ncallbacks: EarlyStopping, ModelCheckpoint]
  I --> J[Test generator]
  J --> K[Predict batches]
  K --> L[Argmax predictions]
  L --> M[Compute metrics\nacc, precision, recall, F1]
  M --> N[Confusion matrix & report]
  N --> O[Paper results]
```

Reproducibility and environment
- Python stack:
  - `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (see `requirements.txt`).
- Determinism: Generators accept `seed` for split reproducibility. Full determinism depends on CUDA/cuDNN and TF settings.
- Checkpoints: Saved in `checkpoint/` or `Checkpoint/` depending on notebook; best models monitored by validation accuracy.

Key code references
- I3D definition: `notebooks/i3d_inception.py` and `lib/i3d_inception/main.py` (function `Inception_Inflated3d`).
- Frame generator: `lib/keras_video/generator.VideoFrameGenerator`.
- Optical flow generator: `lib/keras_video/flow.OpticalFlowGenerator`.
- Helpers: `lib/helpers.py` (class discovery, patterns, and score utilities).
- Example training/evaluation: `notebooks/tari_i3d.ipynb`, `notebooks/tari_vgg16_lstm*.ipynb`.

How this maps to the paper
- Method section: Describe the two pipelines (I3D and VGG16+LSTM), dataset organization, and generator-based sampling.
- Experiment section: Report train/val/test splits, training setup (optimizer, loss, epochs, callbacks), and metrics.
- Results section: Include accuracy, macro-precision/recall/F1, and confusion matrices.
- Discussion: Compare I3D vs VGG16+LSTM at different resolutions; note how higher input resolution improves VGG16+LSTM performance and how I3D captures spatiotemporal features end-to-end.
