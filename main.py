"""
Main training pipeline for spam email classification
Orchestrates the complete machine learning workflow
"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator

def main():
    """Execute the complete spam classification pipeline"""
    
    print("🚀 SPAM EMAIL CLASSIFICATION PROJECT")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Data Loading and Processing
    print("STEP 1: DATA LOADING AND PROCESSING")
    print("-" * 40)
    
    data_processor = DataProcessor()
    
    # Load data (modify path to your file)
    data_file = 'data/raw/spam_ham_dataset.csv'
    df_raw = data_processor.load_and_parse_data(data_file)
    
    # Clean and preprocess data
    df_clean = data_processor.clean_and_preprocess_data(df_raw)
    
    if len(df_clean) < 10:
        print("❌ Insufficient data after cleaning. Using sample dataset.")
        df_clean = data_processor._create_sample_dataset()
        df_clean = data_processor.clean_and_preprocess_data(df_clean)
    
    # Step 2: Text Preprocessing and Feature Extraction
    print("\nSTEP 2: TEXT PREPROCESSING AND FEATURE EXTRACTION")
    print("-" * 40)
    
    feature_extractor = FeatureExtractor()
    
    # Apply advanced text preprocessing
    print("🔤 Applying advanced text preprocessing...")
    df_clean['processed_message'] = df_clean['message'].apply(
        feature_extractor.advanced_text_preprocessing
    )
    
    # Remove empty processed messages
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['processed_message'].str.len() > 3]
    print(f"   Removed {initial_count - len(df_clean)} messages with insufficient content")
    
    # Create TF-IDF features (primary)
    X_tfidf = feature_extractor.create_tfidf_features(
        df_clean['processed_message'].tolist(), max_features=5000
    )
    y = df_clean['target'].values
    
    print(f"📊 Final dataset statistics:")
    print(f"   Total samples: {len(y)}")
    print(f"   Spam samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"   Ham samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    print(f"   Feature dimensions: {X_tfidf.shape[1]}")
    
    # Step 3: Model Training
    print("\nSTEP 3: MODEL TRAINING")
    print("-" * 40)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    trainer.split_data(X_tfidf, y, test_size=0.2)
    
    # Train models with hyperparameter tuning
    nb_model = trainer.train_naive_bayes(tune_hyperparameters=True)
    lr_model = trainer.train_logistic_regression(tune_hyperparameters=True)
    
    # Cross-validation
    cv_results = trainer.cross_validate_models()
    
    # Step 4: Model Evaluation
    print("\nSTEP 4: MODEL EVALUATION")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    
    # Evaluate both models
    nb_metrics = evaluator.evaluate_model(
        nb_model, trainer.X_test, trainer.y_test, 'naive_bayes'
    )
    lr_metrics = evaluator.evaluate_model(
        lr_model, trainer.X_test, trainer.y_test, 'logistic_regression'
    )
    
    # Compare models
    comparison_df = evaluator.compare_models()
    
    # Step 5: Save Models and Results
    print("\nSTEP 5: SAVING MODELS AND RESULTS")
    print("-" * 40)
    
    # Save models
    trainer.save_models()
    
    # Save feature extractors
    joblib.dump(feature_extractor, 'models/feature_extractor.pkl')
    joblib.dump(data_processor, 'models/data_processor.pkl')
    
    # Save evaluation results
    joblib.dump(evaluator.results, 'models/evaluation_results.pkl')
    
    # Save processed data
    df_clean.to_csv('data/processed/processed_spam_data.csv', index=False)
    
    print("   ✅ Saved naive_bayes_optimized.pkl")
    print("   ✅ Saved logistic_regression_optimized.pkl")
    print("   ✅ Saved feature_extractor.pkl")
    print("   ✅ Saved evaluation_results.pkl")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📊 FINAL RESULTS SUMMARY:")
    if comparison_df is not None:
        print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n📁 Generated Files:")
    print(f"   ├── models/naive_bayes_optimized.pkl")
    print(f"   ├── models/logistic_regression_optimized.pkl")
    print(f"   ├── models/feature_extractor.pkl")
    print(f"   ├── models/evaluation_results.pkl")
    print(f"   └── data/processed/processed_spam_data.csv")
    
    print(f"\n🚀 Next Steps:")
    print(f"   Run: streamlit run streamlit_app.py")
    print(f"   Open: http://localhost:8501")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
