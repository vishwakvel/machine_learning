# ML Model Zoo

## What this repository is
`ml_model_zoo/` is a scaffolded machine learning model zoo that organizes a wide range of algorithms into clear, independent folders. Each model has its own workspace with `model.py`, `trainer.py`, `config.py`, `utils.py`, `__init__.py`, and a model-specific `README.md`.

## How it is organized
- `core/`: Shared infrastructure placeholders (base abstractions, metrics, optimizers, losses, activations, data, regularization, initializers, schedulers, and utilities).
- `supervised/`: Regression and classification models grouped by family.
- `unsupervised/`: Clustering, dimensionality reduction, and anomaly detection models.
- `reinforcement_learning/`: Environment definitions and RL algorithm families.
- `deep_learning/`: Engine pieces, layers, and deep model families.
- `ensembles/`: Ensemble learning approaches.
- `tests/`: Task-oriented test directories.
- `benchmarks/`: Benchmark runner and result folders.

## Full model list by category
### Deep Learning
- Bert Style
- Convolutional Neural Network
- Diffusion
- Efficientnet
- GPT Style
- GRU
- Generative Adversarial Network
- LSTM
- Multilayer Perceptron
- RNN
- Resnet
- Seq2seq
- T5 Style
- Temporal Conv Net
- Transformer Encoder
- U-Net
- Variational Autoencoder
- Vgg
- Vision Transformer

### Ensembles
- Bagging
- Blending
- Boosting
- Snapshot Ensemble
- Stacking
- Voting Classifier
- Voting Regressor

### Reinforcement Learning
- Advantage Actor-Critic
- Asynchronous Advantage Actor-Critic
- Cartpole
- Deep Q-Network
- Double Deep Q-Network
- Dueling Deep Q-Network
- Dyna Q
- Gridworld
- Proximal Policy Optimization
- Q Learning
- Reinforce
- Reinforce Baseline
- Sarsa
- Twin Delayed DDPG

### Supervised/Classification
- Adaline
- Catboost Classifier
- Decision Tree Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Kernel Support Vector Machine
- Knn Classifier
- Lightgbm Classifier
- Linear Discriminant Analysis
- Linear Support Vector Machine
- Logistic Regression
- Mlp Classifier
- Naive Bayes Bernoulli
- Naive Bayes Complement
- Naive Bayes Gaussian
- Naive Bayes Multinomial
- Perceptron
- Random Forest Classifier
- Ridge Classifier
- Sgd Classifier
- Xgboost Classifier

### Supervised/Regression
- Bayesian Linear Regression
- Bayesian Ridge
- Catboost Regressor
- Decision Tree Regressor
- Elastic Net
- Extra Trees Regressor
- Gaussian Process Regression
- Gradient Boosting Regressor
- Knn Regressor
- Lasso Regression
- Lightgbm Regressor
- Linear Regression
- Polynomial Regression
- Quantile Regression
- Random Forest Regressor
- Ridge Regression
- Support Vector Regression
- Xgboost Regressor

### Unsupervised/Anomaly Detection
- Autoencoder Anomaly
- Elliptic Envelope
- Isolation Forest
- Local Outlier Factor
- One Class Svm

### Unsupervised/Clustering
- Agglomerative Clustering
- Birch
- Dbscan
- Dirichlet Process Gmm
- Divisive Clustering
- Fuzzy C Means
- Gaussian Mixture Model
- Hdbscan
- K Means
- K Means Plus Plus
- K Medoids
- Mini Batch K Means
- Optics

### Unsupervised/Dimensionality Reduction
- Autoencoders
- Factor Analysis
- Incremental PCA
- Independent Component Analysis
- Isomap
- Kernel PCA
- Locally Linear Embedding
- Non-negative Matrix Factorization
- PCA
- SVD
- Spectral Embedding
- UMAP
- t-SNE

## How to navigate
- Start from task type (`supervised`, `unsupervised`, `reinforcement_learning`, `deep_learning`, `ensembles`).
- Drill into family folders (for example, `tree_based`, `svm`, `transformers`).
- Open a model folder to find the dedicated README and the five placeholder Python files.
- Use `core/` when you need shared reusable foundations instead of model-specific artifacts.
