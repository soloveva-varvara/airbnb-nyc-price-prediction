# Predicting Airbnb NYC Listing Prices

A machine learning pipeline predicting Airbnb listing prices in New York City, using the
[Inside Airbnb NYC 2019](http://insideairbnb.com/) dataset. Covers data exploration, cleaning
and feature engineering (including NLP embeddings for listing names), and model
development across linear (Ridge), tree-based (Random Forest, XGBoost) and neural
network (MLP, Keras) regressors, with a further round of hyperparameter tuning via grid
search and cross-validation.

## Project structure

The pipeline is split across four notebooks, run in order:

| Notebook | Covers |
|---|---|
| `part_1_data_exploration.ipynb` | Initial data exploration and visualisation |
| `part_2_data_cleaning.ipynb` | Cleaning, outlier handling, feature engineering (name embeddings via Word2Vec + GloVe, PCA, one-hot encoding) |
| `part_3_ML_modelling.ipynb` | Train/validation/test split, and initial model comparison (Ridge, Random Forest, XGBoost, multiple MLP variants) |
| `part_4_Model_Fine_Tuning.ipynb` | Grid search and cross-validation fine-tuning of the best-performing models |

```
Airbnb_ml_project/
├── part_1_data_exploration.ipynb
├── part_2_data_cleaning.ipynb
├── part_3_ML_modelling.ipynb
├── part_4_Model_Fine_Tuning.ipynb
├── requirements.txt
└── data/
    ├── raw/            # AB_NYC_2019.csv — original dataset
    ├── processed/       # cleaned/feature-engineered dataframes saved between notebooks
    └── arrays/          # train/val/test splits (.npy), saved after part_3, loaded in part_4
```

## Setup

### 1. Create the environment

```bash
conda create -n airbnb-text python=3.11 -y
conda run -n airbnb-text pip install -r requirements.txt
```

### 2. Install the OpenMP runtime (macOS only)

`xgboost` needs the OpenMP runtime to import, which isn't installable via pip on macOS:

```bash
conda install -n airbnb-text -y llvm-openmp
```

### 3. Register the Jupyter kernel

```bash
conda run -n airbnb-text python -m ipykernel install --user --name airbnb-text --display-name "Python 3.11 (airbnb-text)"
```

In Jupyter, select **"Python 3.11 (airbnb-text)"** as the kernel for each notebook.

### 4. Download GloVe embeddings

`part_2_data_cleaning.ipynb` uses pre-trained GloVe word embeddings (100-dimensional) for the
`name` column. Download and extract them into a `glove/` folder in the project root:

```bash
curl -L -o glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt -d glove/
rm glove.6B.zip
```

## Running

Run the notebooks in order (1 → 4). Each notebook saves the data or arrays the next one
needs into `data/`, so later notebooks can't be run standalone without first running the
ones before them at least once.

## Results

Across both rounds of model development, tree-based ensemble methods (Random Forest,
XGBoost) outperformed both the linear (Ridge) and neural network (MLP) approaches, with
the fine-tuned XGBoost model achieving the best test-set performance. Extensive
hyperparameter tuning of the MLP models did not close this gap, and did so at
substantially higher computational cost — see `part_4_Model_Fine_Tuning.ipynb` for the
full comparison.
