# Core Infrastructure

## What `core/` contains
The `core/` folder centralizes reusable building blocks that many models rely on. Keeping these shared pieces in one place avoids duplicate implementations and gives every model a consistent interface for optimization, losses, metrics, data handling, and utility functions.

## Subfolders and purpose
- `base/`: Abstract base placeholders for common model types (generic model, classifier, regressor, clusterer, transformer).
- `metrics/`: Shared metric placeholders for regression, classification, clustering, and ranking tasks.
- `optimizers/`: Optimizer placeholders such as SGD, Adam, Adagrad, RMSProp, and AdamW.
- `losses/`: Common loss placeholders for supervised and reinforcement learning workflows.
- `activations/`: Activation-related placeholders, including softmax.
- `data/`: Data pipeline placeholders for loading, splitting, preprocessing, and dataset abstractions.
- `regularization/`: Regularization placeholders such as dropout, normalization, and L1/L2 penalties.
- `initializers/`: Parameter initialization placeholders for weights and biases.
- `schedulers/`: Learning-rate scheduler placeholders.
- `utils/`: Shared math, matrix, serialization, randomness, and logging placeholders.

## Files in `core/`
### `base/`
- `base_model.py`
- `base_classifier.py`
- `base_regressor.py`
- `base_cluster.py`
- `base_transformer.py`

### `metrics/`
- `regression_metrics.py`
- `classification_metrics.py`
- `clustering_metrics.py`
- `ranking_metrics.py`

### `optimizers/`
- `sgd.py`
- `adam.py`
- `adagrad.py`
- `rmsprop.py`
- `adamw.py`

### `losses/`
- `regression_losses.py`
- `classification_losses.py`
- `rl_losses.py`

### `activations/`
- `activations.py`
- `softmax.py`

### `data/`
- `data_loader.py`
- `data_splitter.py`
- `preprocessing.py`
- `dataset.py`

### `regularization/`
- `dropout.py`
- `batch_norm.py`
- `layer_norm.py`
- `l1_l2.py`

### `initializers/`
- `weight_init.py`
- `bias_init.py`

### `schedulers/`
- `lr_scheduler.py`
- `reduce_on_plateau.py`

### `utils/`
- `math_utils.py`
- `matrix_ops.py`
- `serialization.py`
- `random.py`
- `logging.py`

## Why this is centralized
Centralizing this layer creates consistency across all model families, makes refactoring easier, and provides one shared foundation that reduces copy-paste code as the zoo grows.
