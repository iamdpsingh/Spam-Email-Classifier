import pickle
import numpy as np
from utils.preprocess import preprocess_text
import os

def load_model_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run model.py first to train the model.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_email(text):
    """Predict if an email is spam or ham"""
    try:
        model, vectorizer = load_model_vectorizer()
        
        if model is None or vectorizer is None:
            return "Error: Model not loaded", 0.0
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        if not processed_text.strip():
            return "Error: No valid text after preprocessing", 0.0
        
        # Transform text using the fitted vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probability for confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.8  # Default confidence for models without predict_proba
        
        # Convert prediction to readable format
        result = "Spam" if prediction == 1 else "Ham"
        
        return result, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error: Prediction failed", 0.0

def batch_predict(texts):
    """Predict multiple emails at once"""
    results = []
    for text in texts:
        result, confidence = predict_email(text)
        results.append({
            'text': text[:50] + "..." if len(text) > 50 else text,
            'prediction': result,
            'confidence': confidence
        })
    return results

# Test function
def test_predictions():
    """Test the prediction function with sample emails"""
    sample_emails = [
        "Congratulations! You've won $1000! Click here to claim now!",
        "Hi John, can we meet for lunch tomorrow at 1 PM?",
        "FREE MONEY! URGENT! Act now and get rich quick!",
        "Meeting reminder: Team standup at 10 AM in conference room."
    ]
    
    print("Testing predictions:")
    for email in sample_emails:
        prediction, confidence = predict_email(email)
        print(f"Email: {email[:50]}...")
        print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
        print("-" * 60)

if __name__ == "__main__":
    test_predictions()
