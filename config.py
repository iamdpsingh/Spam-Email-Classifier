# Configuration file for Spam Email Classifier

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Model Configuration
TFIDF_CONFIG = {
    'max_features': 5000,
    'stop_words': 'english',
    'lowercase': True,
    'max_df': 0.9,
    'min_df': 1,
    'ngram_range': (1, 2)
}

NAIVE_BAYES_CONFIG = {
    'alpha': 0.1
}

LOGISTIC_REGRESSION_CONFIG = {
    'random_state': 42,
    'max_iter': 2000,
    'C': 10.0,
    'class_weight': 'balanced'
}

# Training Configuration
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
