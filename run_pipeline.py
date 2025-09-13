#!/usr/bin/env python3
"""
Complete pipeline runner for Spam Email Classifier
Run this script to execute the entire machine learning pipeline with your dataset
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import process_your_dataset
from src.model_training import train_spam_classifiers
from src.utils import check_system_status, analyze_dataset
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR

def setup_directories():
    """Create necessary directories"""
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def process_dataset(dataset_path):
    """Process the user's dataset"""
    print("\n🧹 PROCESSING YOUR DATASET")
    print("="*50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    # Process the dataset
    cleaned_path = process_your_dataset(dataset_path, PROCESSED_DATA_DIR)
    
    if cleaned_path:
        print(f"✅ Dataset processed successfully!")
        print(f"📁 Cleaned dataset saved to: {cleaned_path}")
        
        # Analyze the processed dataset
        import pandas as pd
        df = pd.read_csv(cleaned_path)
        analyze_dataset(df)
        
        return cleaned_path
    else:
        print("❌ Dataset processing failed")
        return None

def train_models(cleaned_data_path):
    """Train machine learning models"""
    print("\n🤖 TRAINING MACHINE LEARNING MODELS")
    print("="*50)
    
    # Train models
    results = train_spam_classifiers(cleaned_data_path)
    
    if results:
        classifier, comparison_df, detailed_df, best_model = results
        print(f"✅ Model training completed successfully!")
        print(f"🏆 Best performing model: {best_model}")
        
        # Display results summary
        if comparison_df is not None:
            print(f"\n📊 Performance Summary:")
            print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return results
    else:
        print("❌ Model training failed")
        return None

def run_tests():
    """Run basic tests on trained models"""
    print("\n🧪 RUNNING MODEL TESTS")
    print("="*30)
    
    # Test samples for various scenarios
    test_samples = [
        ("FREE! Win money now! Click here immediately!", "Expected: SPAM"),
        ("Hi, the meeting is scheduled for 2 PM tomorrow.", "Expected: HAM"),
        ("URGENT! Your account needs immediate verification!", "Expected: SPAM"),
        ("Thanks for sending the quarterly report on time.", "Expected: HAM"),
        ("Congratulations! You've won a million dollars! Claim now!", "Expected: SPAM"),
        ("Please review the attached project proposal when convenient.", "Expected: HAM")
    ]
    
    # Load best model
    model_files = ['naive_bayes_model.pkl', 'logistic_regression_model.pkl']
    model = None
    
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                print(f"✅ Loaded model: {model_file}")
            break
    
    if model:
        print("\n🔍 Testing model predictions:")
        print("-" * 70)
        
        correct_predictions = 0
        
        for i, (text, expected) in enumerate(test_samples, 1):
            try:
                prediction = model.predict([text])[0]
                probabilities = model.predict_proba([text])[0]
                
                pred_label = "SPAM" if prediction == 1 else "HAM"
                confidence = probabilities[prediction] * 100
                
                # Check if prediction matches expectation
                expected_label = expected.split(": ")[1]
                is_correct = pred_label == expected_label
                correct_predictions += is_correct
                
                status = "✅" if is_correct else "❌"
                print(f"{status} Test {i}: {pred_label} ({confidence:.1f}% confidence)")
                print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"   {expected}")
                print()
                
            except Exception as e:
                print(f"❌ Error predicting test {i}: {e}")
        
        accuracy = correct_predictions / len(test_samples) * 100
        print(f"📊 Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_samples)} correct)")
        
        if accuracy >= 80:
            print("🎉 Excellent! Model is performing well on test samples.")
        elif accuracy >= 60:
            print("👍 Good performance, but could be improved with more training data.")
        else:
            print("⚠️ Model performance could be better. Consider improving your dataset.")
    
    else:
        print("❌ No trained models found for testing")

def launch_streamlit():
    """Launch Streamlit application"""
    print("\n🌐 LAUNCHING STREAMLIT APPLICATION")
    print("="*40)
    print("📱 The application will be available at: http://localhost:8501")
    print("🔧 Use Ctrl+C to stop the application")
    print("💡 The web interface provides full functionality including:")
    print("   - Email classification")
    print("   - Dataset processing")
    print("   - Model training")
    print("   - Performance analysis")
    print("   - Interactive visualizations")
    print()
    
    # Launch Streamlit
    try:
        os.system("streamlit run streamlit_app.py")
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Spam Email Classifier Pipeline for Your Dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to your dataset file')
    parser.add_argument('--skip-processing', action='store_true', help='Skip dataset processing (use existing cleaned data)')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-streamlit', action='store_true', help='Skip Streamlit launch')
    parser.add_argument('--run-tests', action='store_true', help='Run model tests after training')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 SPAM EMAIL CLASSIFIER - PIPELINE FOR YOUR DATASET")
    print("="*70)
    print(f"📊 Input dataset: {args.dataset}")
    print()
    
    # Setup directories
    setup_directories()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        print("Please provide a valid path to your dataset file.")
        return
    
    # Dataset processing
    cleaned_data_path = None
    
    if not args.skip_processing:
        cleaned_data_path = process_dataset(args.dataset)
        if not cleaned_data_path:
            print("❌ Pipeline failed at data processing stage")
            return
    else:
        # Look for existing cleaned dataset
        cleaned_data_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_dataset.csv')
        if not os.path.exists(cleaned_data_path):
            print("❌ No cleaned dataset found. Run without --skip-processing first.")
            return
        print(f"✅ Using existing cleaned dataset: {cleaned_data_path}")
    
    # Model training
    training_results = None
    if not args.skip_training:
        training_results = train_models(cleaned_data_path)
        if not training_results:
            print("❌ Pipeline failed at model training stage")
            return
    else:
        print("⏩ Skipping model training (using existing models)")
    
    # Run tests if requested
    if args.run_tests:
        run_tests()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("📋 What was accomplished:")
    print("   ✅ Dataset processed and cleaned")
    if not args.skip_training:
        print("   ✅ Machine learning models trained")
    print("   ✅ System ready for use")
    
    if training_results:
        _, comparison_df, _, best_model = training_results
        if comparison_df is not None:
            best_accuracy = comparison_df['Accuracy'].max()
            best_precision = comparison_df['Precision'].max()
            print(f"\n🏆 Best Model Performance:")
            print(f"   Model: {best_model}")
            print(f"   Accuracy: {best_accuracy:.1%}")
            print(f"   Precision: {best_precision:.1%}")
    
    print(f"\n📁 Files created:")
    print(f"   📊 Processed dataset: {cleaned_data_path}")
    print(f"   🤖 Trained models: {MODELS_DIR}/")
    print(f"   📈 Results: {RESULTS_DIR}/")
    
    # Launch Streamlit
    if not args.skip_streamlit:
        print(f"\n🌐 Launching web application...")
        launch_streamlit()
    else:
        print(f"\n💡 To launch the web application, run:")
        print(f"   streamlit run streamlit_app.py")
        print(f"\n✨ Your spam email classifier is ready to use!")

if __name__ == "__main__":
    main()
