import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from utils.preprocess import preprocess_text
import warnings
import os
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('data/spamhamdata.csv', encoding='latin-1')
        
        # Handle different column names
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']].copy()
            df.columns = ['label', 'text']
        elif 'Category' in df.columns and 'Message' in df.columns:
            df = df[['Category', 'Message']].copy()
            df.columns = ['label', 'text']
        
        # Clean the data
        df = df.dropna()
        df['label'] = df['label'].str.lower()
        
        print(f"Dataset loaded successfully with {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def generate_analysis_charts(df):
    """Generate comprehensive analysis charts using matplotlib"""
    try:
        # Create output directory
        os.makedirs('static/images', exist_ok=True)
        
        # Create comprehensive analysis
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Top spam words
        plt.subplot(3, 3, 1)
        spam_text = ' '.join(df[df['label'] == 'spam']['text']).lower()
        spam_words = re.findall(r'\b[a-zA-Z]{3,}\b', spam_text)
        spam_counter = Counter(spam_words).most_common(15)
        if spam_counter:
            words, counts = zip(*spam_counter)
            plt.barh(range(len(words)), counts, color='lightcoral', alpha=0.8)
            plt.yticks(range(len(words)), words)
            plt.title('Top 15 Words in SPAM Emails', fontsize=14, fontweight='bold')
            plt.xlabel('Frequency')
        
        # 2. Top ham words
        plt.subplot(3, 3, 2)
        ham_text = ' '.join(df[df['label'] == 'ham']['text']).lower()
        ham_words = re.findall(r'\b[a-zA-Z]{3,}\b', ham_text)
        ham_counter = Counter(ham_words).most_common(15)
        if ham_counter:
            words, counts = zip(*ham_counter)
            plt.barh(range(len(words)), counts, color='lightblue', alpha=0.8)
            plt.yticks(range(len(words)), words)
            plt.title('Top 15 Words in HAM Emails', fontsize=14, fontweight='bold')
            plt.xlabel('Frequency')
        
        # 3. Label distribution
        plt.subplot(3, 3, 3)
        df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title('Email Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        # 4. Text length comparison
        plt.subplot(3, 3, 4)
        df['text_length'] = df['text'].str.len()
        spam_lengths = df[df['label'] == 'spam']['text_length']
        ham_lengths = df[df['label'] == 'ham']['text_length']
        plt.boxplot([ham_lengths, spam_lengths], labels=['Ham', 'Spam'])
        plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Characters')
        
        # 5. Exclamation marks
        plt.subplot(3, 3, 5)
        df['exclamation_count'] = df['text'].str.count('!')
        spam_excl = df[df['label'] == 'spam']['exclamation_count']
        ham_excl = df[df['label'] == 'ham']['exclamation_count']
        plt.boxplot([ham_excl, spam_excl], labels=['Ham', 'Spam'])
        plt.title('Exclamation Marks Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count')
        
        # 6. Capital letters ratio
        plt.subplot(3, 3, 6)
        df['caps_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        spam_caps = df[df['label'] == 'spam']['caps_ratio']
        ham_caps = df[df['label'] == 'ham']['caps_ratio']
        plt.boxplot([ham_caps, spam_caps], labels=['Ham', 'Spam'])
        plt.title('Capital Letters Ratio', fontsize=14, fontweight='bold')
        plt.ylabel('Ratio')
        
        # 7. Word count distribution
        plt.subplot(3, 3, 7)
        df['word_count'] = df['text'].str.split().str.len()
        plt.hist([df[df['label'] == 'ham']['word_count'], df[df['label'] == 'spam']['word_count']], 
                bins=30, alpha=0.7, label=['Ham', 'Spam'], color=['lightblue', 'lightcoral'])
        plt.title('Word Count Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 8. Average word length
        plt.subplot(3, 3, 8)
        df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
        spam_avg_len = df[df['label'] == 'spam']['avg_word_length']
        ham_avg_len = df[df['label'] == 'ham']['avg_word_length']
        plt.boxplot([ham_avg_len, spam_avg_len], labels=['Ham', 'Spam'])
        plt.title('Average Word Length', fontsize=14, fontweight='bold')
        plt.ylabel('Characters per Word')
        
        # 9. Question marks
        plt.subplot(3, 3, 9)
        df['question_count'] = df['text'].str.count('\?')
        spam_quest = df[df['label'] == 'spam']['question_count']
        ham_quest = df[df['label'] == 'ham']['question_count']
        plt.boxplot([ham_quest, spam_quest], labels=['Ham', 'Spam'])
        plt.title('Question Marks Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('static/images/text_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate a separate word frequency comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Spam word frequency
        if spam_counter:
            words, counts = zip(*spam_counter[:20])
            ax1.barh(range(len(words)), counts, color='red', alpha=0.7)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_title('Most Frequent Words in SPAM Emails', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Frequency')
        
        # Ham word frequency
        if ham_counter:
            words, counts = zip(*ham_counter[:20])
            ax2.barh(range(len(words)), counts, color='green', alpha=0.7)
            ax2.set_yticks(range(len(words)))
            ax2.set_yticklabels(words)
            ax2.set_title('Most Frequent Words in HAM Emails', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('static/images/word_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Analysis charts generated!")
        print("   - static/images/text_analysis.png")
        print("   - static/images/word_frequency.png")
        
    except Exception as e:
        print(f"Error generating analysis charts: {e}")

def train_models():
    """Train multiple models and select the best one"""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate analysis charts
    generate_analysis_charts(df)
    
    # Preprocess text
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Prepare features and labels
    X = df['processed_text']
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit vectorizer
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Models to train
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_vect, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vect)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Final evaluation with best model
    y_pred_best = best_model.predict(X_test_vect)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot model comparison
    plt.figure(figsize=(10, 6))
    models_names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)
    
    # Add accuracy values on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save best model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'feature_count': X_train_vect.shape[1],
        'training_samples': len(X_train)
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nModel saved successfully!")
    print(f"Files saved: model.pkl, vectorizer.pkl, model_metadata.pkl")
    print(f"Charts saved in: static/images/")
    
    return best_model, vectorizer

if __name__ == "__main__":
    train_models()
