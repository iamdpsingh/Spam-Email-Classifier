import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_path):
    """Load a trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_email(text, model):
    """Predict if an email is spam or ham"""
    try:
        if model is None:
            return None
            
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probabilities[prediction] * 100
        
        return {
            'prediction': label,
            'confidence': confidence,
            'spam_probability': probabilities[1] * 100,
            'ham_probability': probabilities[0] * 100
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def validate_dataset(df):
    """Validate dataset format and quality"""
    issues = []
    
    # Check if DataFrame is valid
    if df is None or df.empty:
        issues.append("Dataset is empty or None")
        return issues
    
    # Check required columns
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if 'text' in df.columns:
        missing_text = df['text'].isna().sum()
        if missing_text > 0:
            issues.append(f"Missing text values: {missing_text}")
    
    if 'label' in df.columns:
        missing_labels = df['label'].isna().sum()
        if missing_labels > 0:
            issues.append(f"Missing label values: {missing_labels}")
    
    # Check label format
    if 'label' in df.columns:
        unique_labels = df['label'].unique()
        valid_labels = {'spam', 'ham', 0, 1, '0', '1'}
        invalid_labels = [label for label in unique_labels if label not in valid_labels]
        if invalid_labels:
            issues.append(f"Invalid/unusual labels found: {invalid_labels}")
    
    # Check dataset size
    if len(df) < 10:
        issues.append("Dataset is very small (< 10 samples) - may affect model performance")
    
    # Check text quality
    if 'text' in df.columns:
        empty_texts = (df['text'].astype(str).str.strip() == '').sum()
        if empty_texts > 0:
            issues.append(f"Empty text entries: {empty_texts}")
        
        short_texts = (df['text'].astype(str).str.len() < 10).sum()
        if short_texts > len(df) * 0.5:
            issues.append(f"Many very short texts ({short_texts}/{len(df)}) - may affect quality")
    
    # Report results
    if issues:
        print("⚠️ Dataset validation issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Dataset validation passed")
        return True

def create_confusion_matrix_plot(y_true, y_pred, model_name, save_path=None):
    """Create confusion matrix visualization"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix plot saved: {save_path}")
        
        plt.show()
        return True
    except Exception as e:
        print(f"❌ Error creating confusion matrix plot: {e}")
        return False

def create_performance_comparison_plot(results_df, save_path=None):
    """Create model performance comparison plot"""
    try:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            values = [model_data[metric].iloc[0] for metric in metrics if metric in model_data.columns]
            available_metrics = [metric for metric in metrics if metric in model_data.columns]
            
            fig.add_trace(go.Bar(
                name=model,
                x=available_metrics,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Performance comparison plot saved: {save_path}")
        
        return fig
    except Exception as e:
        print(f"❌ Error creating performance plot: {e}")
        return None

def analyze_dataset(df):
    """Analyze dataset characteristics"""
    if df is None or df.empty:
        print("❌ Cannot analyze empty dataset")
        return None
    
    print("📊 DATASET ANALYSIS")
    print("="*40)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'text' in df.columns and 'label' in df.columns:
        # Text analysis
        print(f"\n📝 Text Analysis:")
        text_lengths = df['text'].astype(str).str.len()
        print(f"   Average text length: {text_lengths.mean():.1f} characters")
        print(f"   Min text length: {text_lengths.min()} characters")
        print(f"   Max text length: {text_lengths.max()} characters")
        print(f"   Median text length: {text_lengths.median():.1f} characters")
        
        # Label analysis
        print(f"\n🏷️ Label Analysis:")
        label_counts = df['label'].value_counts()
        print(f"   Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"     {label}: {count} ({percentage:.1f}%)")
        
        # Class balance
        if len(label_counts) > 1:
            balance_ratio = label_counts.max() / label_counts.min()
            print(f"   Class balance ratio: {balance_ratio:.2f}")
            if balance_ratio > 3:
                print("   ⚠️ Dataset is imbalanced - consider balancing techniques")
        
        # Sample examples
        print(f"\n📋 Sample Examples:")
        for label in df['label'].unique():
            sample = df[df['label'] == label]['text'].iloc[0]
            print(f"   {label}: '{sample[:100]}{'...' if len(sample) > 100 else ''}'")
    
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'text_stats': text_lengths.describe().to_dict() if 'text' in df.columns else None,
        'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else None
    }

def check_system_status():
    """Check if all required directories and files exist"""
    from config import MODELS_DIR, RESULTS_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    print("🔍 SYSTEM STATUS CHECK")
    print("="*30)
    
    directories = {
        'Data Directory': DATA_DIR,
        'Raw Data': RAW_DATA_DIR,
        'Processed Data': PROCESSED_DATA_DIR,
        'Models': MODELS_DIR,
        'Results': RESULTS_DIR
    }
    
    all_good = True
    
    for name, path in directories.items():
        if os.path.exists(path):
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"✅ {name}: {path} ({file_count} files)")
        else:
            print(f"❌ {name}: {path} (missing)")
            all_good = False
    
    # Check for trained models
    model_files = ['naive_bayes_model.pkl', 'logistic_regression_model.pkl']
    trained_models = []
    
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_path):
            trained_models.append(model_file)
    
    if trained_models:
        print(f"🤖 Trained models: {trained_models}")
    else:
        print("⚠️ No trained models found - run training first")
    
    return all_good

def get_project_info():
    """Get comprehensive project information"""
    info = {
        'project_name': 'Spam Email Classifier',
        'version': '1.0.0',
        'algorithms': ['Naive Bayes', 'Logistic Regression'],
        'features': [
            'Universal Dataset Cleaner',
            'TF-IDF Feature Extraction',
            'Interactive Web Interface',
            'Real-time Predictions',
            'Performance Visualizations',
            'Model Comparison'
        ],
        'technologies': [
            'Python 3.8+',
            'Scikit-learn',
            'Streamlit',
            'Plotly',
            'Pandas',
            'NumPy'
        ],
        'metrics': [
            'Accuracy',
            'Precision', 
            'Recall',
            'F1-Score',
            'ROC-AUC',
            'Confusion Matrix'
        ]
    }
    
    return info
