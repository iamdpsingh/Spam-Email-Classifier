from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from predict import predict_email

app = Flask(__name__)

# Ensure static/images directory exists
os.makedirs('static/images', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    confidence = ""
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        if email_text.strip():
            prediction, conf = predict_email(email_text)
            result = prediction
            confidence = f"Confidence: {conf:.2%}"
        else:
            result = "Please enter email text"
    
    return render_template('index.html', result=result, confidence=confidence)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()
    if data and 'text' in data:
        prediction, confidence = predict_email(data['text'])
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence)
        })
    return jsonify({'error': 'No text provided'}), 400

@app.route('/analytics')
def view_analytics():
    """Route to view generated analysis charts"""
    text_analysis_exists = os.path.exists('static/images/text_analysis.png')
    word_frequency_exists = os.path.exists('static/images/word_frequency.png')
    confusion_matrix_exists = os.path.exists('static/images/confusion_matrix.png')
    model_comparison_exists = os.path.exists('static/images/model_comparison.png')
    
    return render_template('analytics.html',
                         text_analysis_exists=text_analysis_exists,
                         word_frequency_exists=word_frequency_exists,
                         confusion_matrix_exists=confusion_matrix_exists,
                         model_comparison_exists=model_comparison_exists)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
        print("Model files not found. Please run model.py first to train the model.")
        print("\nTo train the model, run: python model.py")
    else:
        print("✅ Model files found!")
        print("🚀 Starting Flask application...")
        print("📊 Access analytics at: http://localhost:5000/analytics")
        app.run(debug=True, host='0.0.0.0', port=5000)
