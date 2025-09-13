import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TFIDF_CONFIG, NAIVE_BAYES_CONFIG, LOGISTIC_REGRESSION_CONFIG, TEST_SIZE, RANDOM_STATE, MODELS_DIR, RESULTS_DIR

class SpamClassifier:
    """Spam classification model using Naive Bayes and Logistic Regression"""
    
    def __init__(self):
        self.models = {
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(**TFIDF_CONFIG)),
                ('classifier', MultinomialNB(**NAIVE_BAYES_CONFIG))
            ]),
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(**TFIDF_CONFIG)),
                ('classifier', LogisticRegression(**LOGISTIC_REGRESSION_CONFIG))
            ])
        }
        self.results = {}
        self.training_data = None
    
    def load_dataset(self, data_path):
        """Load and prepare dataset"""
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded: {df.shape}")
            
            # Check required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must have 'text' and 'label' columns")
            
            # Show dataset info
            print(f"📊 Label distribution:")
            print(df['label'].value_counts())
            print(f"\n📈 Sample lengths:")
            print(f"   Average text length: {df['text'].str.len().mean():.1f} chars")
            print(f"   Min text length: {df['text'].str.len().min()} chars")
            print(f"   Max text length: {df['text'].str.len().max()} chars")
            
            self.training_data = df
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Extract features and labels
        X = df['text']
        y = df['label'].map({'spam': 1, 'ham': 0})
        
        # Check for any unmapped labels
        if y.isna().any():
            print("⚠️ Warning: Some labels could not be mapped")
            print("Unmapped labels:", df.loc[y.isna(), 'label'].unique())
            
            # Remove unmapped labels
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            print(f"Removed {(~mask).sum()} samples with unmapped labels")
        
        return X, y
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train both models and evaluate performance"""
        
        print(f"\n🎯 Training Models")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Training label distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"🤖 Training {model_name}")
            print(f"{'='*60}")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Handle ROC-AUC calculation
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    roc_auc = 0.5  # Default for single class
                    print("⚠️ ROC-AUC could not be calculated (single class?)")
                
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Store results
                self.results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': conf_matrix,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                # Print results
                print(f"✅ Training completed!")
                print(f"📊 Performance Metrics:")
                print(f"   Accuracy:  {accuracy:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall:    {recall:.4f}")
                print(f"   F1-Score:  {f1:.4f}")
                print(f"   ROC-AUC:   {roc_auc:.4f}")
                
                print(f"\n🎯 Confusion Matrix:")
                if conf_matrix.shape == (2, 2):
                    print(f"   TN={conf_matrix[0,0]}, FP={conf_matrix[0,1]}")
                    print(f"   FN={conf_matrix[1,0]}, TP={conf_matrix[1,1]}")
                else:
                    print(f"   {conf_matrix}")
                
                print(f"\n📋 Detailed Classification Report:")
                print(classification_report(y_test, y_pred))
                
                # Save the model
                model_filename = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 Model saved: {model_filename}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
    
    def compare_models(self):
        """Compare model performance"""
        if not self.results:
            print("❌ No models trained yet!")
            return None, None
            
        print(f"\n{'='*80}")
        print("🏆 MODEL PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'ROC-AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model by F1-score
        best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df['F1-Score'].max()
        
        print(f"\n🥇 Best Model: {best_model_name}")
        print(f"🎯 Best F1-Score: {best_f1:.4f}")
        
        # Save comparison results
        comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"💾 Comparison saved: {comparison_path}")
        
        return comparison_df, best_model_name
    
    def save_detailed_results(self):
        """Save detailed evaluation results"""
        if not self.results:
            return None
            
        detailed_data = []
        for model_name, results in self.results.items():
            cm = results['confusion_matrix']
            
            # Handle different confusion matrix shapes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                # Single class case
                tp = cm[0, 0] if len(cm) > 0 else 0
                tn = fp = fn = 0
                specificity = npv = 1.0
            
            detailed_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1'],
                'ROC_AUC': results['roc_auc'],
                'True_Positives': tp,
                'False_Positives': fp,
                'True_Negatives': tn,
                'False_Negatives': fn,
                'Specificity': specificity,
                'NPV': npv
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Save detailed results
        detailed_path = os.path.join(RESULTS_DIR, 'detailed_evaluation.csv')
        detailed_df.to_csv(detailed_path, index=False)
        print(f"💾 Detailed evaluation saved: {detailed_path}")
        
        return detailed_df
    
    def test_model_predictions(self, model_name=None):
        """Test model with sample predictions"""
        if not self.results:
            print("❌ No trained models available")
            return
        
        # Use best model if none specified
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        
        if model_name not in self.results:
            print(f"❌ Model '{model_name}' not found")
            return
        
        model = self.results[model_name]['model']
        
        # Test samples
        test_samples = [
            "FREE! Win money now! Click here immediately!",
            "Hi, the meeting is scheduled for 2 PM tomorrow.",
            "URGENT! Your account needs immediate verification!",
            "Thanks for sending the quarterly report.",
            "Congratulations! You've won a million dollars!",
            "Please review the attached project proposal."
        ]
        
        print(f"\n🧪 Testing {model_name} with sample predictions:")
        print("="*70)
        
        for i, sample in enumerate(test_samples, 1):
            try:
                prediction = model.predict([sample])[0]
                probabilities = model.predict_proba([sample])[0]
                
                pred_label = "SPAM" if prediction == 1 else "HAM"
                confidence = probabilities[prediction] * 100
                
                print(f"{i}. Text: '{sample[:50]}{'...' if len(sample) > 50 else ''}'")
                print(f"   Prediction: {pred_label} (Confidence: {confidence:.1f}%)")
                print()
                
            except Exception as e:
                print(f"   Error predicting sample {i}: {e}")

def train_spam_classifiers(data_path):
    """Main training function for your dataset"""
    
    print("🚀 SPAM EMAIL CLASSIFIER TRAINING")
    print("="*60)
    
    # Initialize classifier
    classifier = SpamClassifier()
    
    # Load your dataset
    df = classifier.load_dataset(data_path)
    if df is None:
        return None
    
    # Prepare data
    X, y = classifier.prepare_data(df)
    
    if len(X) == 0:
        print("❌ No valid data for training")
        return None
    
    # Check for minimum samples
    if len(X) < 4:
        print("❌ Not enough samples for train/test split (minimum 4 required)")
        return None
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=min(TEST_SIZE, 0.5), random_state=RANDOM_STATE, 
            stratify=y if len(y.unique()) > 1 else None
        )
    except ValueError as e:
        # If stratify fails, split without stratification
        print(f"⚠️ Stratified split failed, using random split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=min(TEST_SIZE, 0.5), random_state=RANDOM_STATE
        )
    
    # Train models
    classifier.train_models(X_train, X_test, y_train, y_test)
    
    if not classifier.results:
        print("❌ No models were successfully trained")
        return None
    
    # Compare models and save results
    comparison_df, best_model = classifier.compare_models()
    detailed_df = classifier.save_detailed_results()
    
    # Test model predictions
    classifier.test_model_predictions()
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"🎯 Best Model: {best_model}")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"📁 Models saved to: {MODELS_DIR}")
    
    return classifier, comparison_df, detailed_df, best_model

if __name__ == "__main__":
    # Example usage with your dataset
    data_path = "data/processed/cleaned_dataset.csv"
    results = train_spam_classifiers(data_path)
