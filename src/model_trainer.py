"""
Model training module with optimized hyperparameters for spam classification
"""
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

class ModelTrainer:
    def __init__(self, random_state=42):
        """Initialize model trainer"""
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        print(f"📊 Splitting data: {len(y)} total samples")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        
        print(f"   Training set: {len(self.y_train)} samples")
        print(f"   Test set: {len(self.y_test)} samples")
        print(f"   Training distribution: {np.bincount(self.y_train)}")
        print(f"   Test distribution: {np.bincount(self.y_test)}")
    
    def train_naive_bayes(self, tune_hyperparameters=True):
        """Train Naive Bayes with optional hyperparameter tuning"""
        print("🤖 Training Naive Bayes model...")
        
        if tune_hyperparameters:
            print("   🔧 Tuning hyperparameters...")
            
            # Grid search for best alpha
            param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
            
            nb = MultinomialNB()
            grid_search = GridSearchCV(
                nb, param_grid, cv=5, scoring='f1', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_models['naive_bayes'] = grid_search.best_estimator_
            print(f"   ✅ Best alpha: {grid_search.best_params_['alpha']}")
            print(f"   ✅ Best CV F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            nb = MultinomialNB(alpha=1.0)
            nb.fit(self.X_train, self.y_train)
            self.best_models['naive_bayes'] = nb
        
        return self.best_models['naive_bayes']
    
    def train_logistic_regression(self, tune_hyperparameters=True):
        """Train Logistic Regression with optional hyperparameter tuning"""
        print("🤖 Training Logistic Regression model...")
        
        if tune_hyperparameters:
            print("   🔧 Tuning hyperparameters...")
            
            # Grid search for best C and solver
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=self.random_state)
            grid_search = GridSearchCV(
                lr, param_grid, cv=5, scoring='f1',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_models['logistic_regression'] = grid_search.best_estimator_
            print(f"   ✅ Best C: {grid_search.best_params_['C']}")
            print(f"   ✅ Best solver: {grid_search.best_params_['solver']}")
            print(f"   ✅ Best CV F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Use optimized default parameters
            lr = LogisticRegression(
                C=10.0, max_iter=1000, solver='liblinear',
                random_state=self.random_state
            )
            lr.fit(self.X_train, self.y_train)
            self.best_models['logistic_regression'] = lr
        
        return self.best_models['logistic_regression']
    
    def cross_validate_models(self):
        """Perform cross-validation on trained models"""
        print("🔄 Performing cross-validation...")
        
        cv_results = {}
        
        for name, model in self.best_models.items():
            scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=5, scoring='f1'
            )
            
            cv_results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }
            
            print(f"   {name}: F1 = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_models(self, path='models/'):
        """Save trained models and vectorizers"""
        print("💾 Saving models...")
        
        for name, model in self.best_models.items():
            filename = f"{path}{name}_optimized.pkl"
            joblib.dump(model, filename)
            print(f"   ✅ Saved {name} to {filename}")
