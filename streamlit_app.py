"""
Professional Streamlit application for spam email classification
User interface with comprehensive features
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .spam-alert {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        margin: 1rem 0;
    }
    
    .ham-safe {
        background: linear-gradient(135deg, #56CCF2, #2F80ED);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(86, 204, 242, 0.4);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalSpamApp:
    def __init__(self):
        """Initialize the professional spam classifier app"""
        self.load_models_and_processors()
        
    def load_models_and_processors(self):
        """Load all trained models and processors"""
        try:
            # Load models
            self.nb_model = joblib.load('models/naive_bayes_optimized.pkl')
            self.lr_model = joblib.load('models/logistic_regression_optimized.pkl')
            
            # Load processors
            self.feature_extractor = joblib.load('models/feature_extractor.pkl')
            self.data_processor = joblib.load('models/data_processor.pkl')
            
            # Load evaluation results
            self.evaluation_results = joblib.load('models/evaluation_results.pkl')
            
            self.models_loaded = True
            
        except Exception as e:
            st.error(f"🚨 Error loading models: {str(e)}")
            st.error("Please run 'python main.py' first to train the models!")
            self.models_loaded = False
    
    def predict_email(self, text):
        """Make prediction using both models"""
        if not self.models_loaded:
            return None
        
        try:
            # Preprocess text
            processed_text = self.feature_extractor.advanced_text_preprocessing(text)
            
            # Extract features
            features = self.feature_extractor.transform_tfidf([processed_text])
            
            # Make predictions
            predictions = {}
            
            # Naive Bayes
            nb_pred = self.nb_model.predict(features)[0]
            nb_proba = self.nb_model.predict_proba(features)[0]
            
            predictions['Naive Bayes'] = {
                'prediction': nb_pred,
                'probability_ham': nb_proba[0],
                'probability_spam': nb_proba[1],
                'confidence': max(nb_proba)
            }
            
            # Logistic Regression
            lr_pred = self.lr_model.predict(features)[0]
            lr_proba = self.lr_model.predict_proba(features)[0]
            
            predictions['Logistic Regression'] = {
                'prediction': lr_pred,
                'probability_ham': lr_proba[0],
                'probability_spam': lr_proba[1],
                'confidence': max(lr_proba)
            }
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def display_prediction_results(self, predictions, email_text):
        """Display prediction results with professional styling"""
        if not predictions:
            return
        
        st.markdown("### 🎯 Classification Results")
        
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        for idx, (model_name, result) in enumerate(predictions.items()):
            with col1 if idx == 0 else col2:
                st.markdown(f"#### {model_name}")
                
                # Display result
                if result['prediction'] == 1:
                    st.markdown(
                        '<div class="spam-alert">🚨 SPAM DETECTED</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="ham-safe">✅ LEGITIMATE EMAIL</div>',
                        unsafe_allow_html=True
                    )
                
                # Probability bars
                st.markdown("**Probability Breakdown:**")
                
                col_ham, col_spam = st.columns(2)
                with col_ham:
                    st.metric(
                        "Ham (Safe)",
                        f"{result['probability_ham']:.1%}",
                        delta=None
                    )
                    st.progress(result['probability_ham'])
                
                with col_spam:
                    st.metric(
                        "Spam (Danger)",
                        f"{result['probability_spam']:.1%}",
                        delta=None
                    )
                    st.progress(result['probability_spam'])
                
                # Confidence score
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")
        
        # Ensemble prediction
        st.markdown("### 🤖 Ensemble Prediction")
        spam_votes = sum(1 for pred in predictions.values() if pred['prediction'] == 1)
        
        if spam_votes >= 1:  # If any model says spam
            st.markdown(
                '<div class="spam-alert">⚠️ FINAL VERDICT: SPAM - Exercise Caution</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ham-safe">✅ FINAL VERDICT: SAFE EMAIL</div>',
                unsafe_allow_html=True
            )
    
    def display_model_performance(self):
        """Display comprehensive model performance metrics"""
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            st.warning("No evaluation results available")
            return
        
        st.markdown("## 📊 Model Performance Analysis")
        
        # Performance metrics table
        metrics_data = []
        for model_name, metrics in self.evaluation_results.items():
            metrics_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.1%}",
                'Precision': f"{metrics['precision']:.1%}",
                'Recall': f"{metrics['recall']:.1%}",
                'F1-Score': f"{metrics['f1_score']:.1%}"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Performance comparison chart
        metrics_for_chart = []
        for model_name, metrics in self.evaluation_results.items():
            metrics_for_chart.extend([
                {'Model': model_name.replace('_', ' ').title(), 'Metric': 'Accuracy', 'Score': metrics['accuracy']},
                {'Model': model_name.replace('_', ' ').title(), 'Metric': 'Precision', 'Score': metrics['precision']},
                {'Model': model_name.replace('_', ' ').title(), 'Metric': 'Recall', 'Score': metrics['recall']},
                {'Model': model_name.replace('_', ' ').title(), 'Metric': 'F1-Score', 'Score': metrics['f1_score']}
            ])
        
        df_chart = pd.DataFrame(metrics_for_chart)
        
        fig = px.bar(
            df_chart, 
            x='Metric', 
            y='Score', 
            color='Model',
            title='Model Performance Comparison',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.markdown("### 🎯 Confusion Matrices")
        col1, col2 = st.columns(2)
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            with col1 if idx == 0 else col2:
                st.markdown(f"#### {model_name.replace('_', ' ').title()}")
                
                cm = metrics['confusion_matrix']
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    ax=ax
                )
                ax.set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                st.pyplot(fig)
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">Spam Email Classifier</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Machine Learning for Email Security</p>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.stop()
        
        # Sidebar with model info
        with st.sidebar:
            st.markdown("## 🔬 Model Information")
            
            st.markdown("### 🤖 Algorithms Used")
            st.write("• **Naive Bayes** (MultinomialNB)")
            st.write("• **Logistic Regression**")
            
            st.markdown("### 📊 Performance Metrics")
            if hasattr(self, 'evaluation_results'):
                for model_name, metrics in self.evaluation_results.items():
                    st.markdown(f"**{model_name.replace('_', ' ').title()}:**")
                    st.write(f"Accuracy: {metrics['accuracy']:.1%}")
                    st.write(f"F1-Score: {metrics['f1_score']:.1%}")
                    st.write("---")
            
            st.markdown("### 🛠️ Features")
            st.write("• Advanced text preprocessing")
            st.write("• TF-IDF feature extraction")
            st.write("• Hyperparameter optimization")
            st.write("• Real-time classification")
            st.write("• Ensemble predictions")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📧 Email Classifier", 
            "📁 File Upload", 
            "📊 Performance Dashboard",
            "🧪 Batch Testing"
        ])
        
        with tab1:
            st.markdown("## ✍️ Email Text Classification")
            
            # Email input
            email_text = st.text_area(
                "Enter email content to classify:",
                height=200,
                placeholder="Paste your email content here...\n\nExample:\nFREE! You've won $1000! Click here now to claim your prize!",
                help="Enter the complete email text including subject and body"
            )
            
            # Classification button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                classify_btn = st.button("🔍 Classify Email", type="primary", use_container_width=True)
            
            if classify_btn and email_text.strip():
                with st.spinner("🤖 Analyzing email content..."):
                    predictions = self.predict_email(email_text)
                    if predictions:
                        self.display_prediction_results(predictions, email_text)
                        
                        # Show processed text
                        with st.expander("🔍 View Processed Text"):
                            processed = self.feature_extractor.advanced_text_preprocessing(email_text)
                            st.code(processed, language='text')
            
            elif classify_btn:
                st.warning("⚠️ Please enter email text to classify!")
        
        with tab2:
            st.markdown("## 📁 File Upload Classification")
            
            uploaded_file = st.file_uploader(
                "Upload email file (.txt format)",
                type=['txt'],
                help="Upload a text file containing email content"
            )
            
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    
                    st.markdown("### 📄 File Content Preview")
                    st.text_area("Content:", content[:1000] + "..." if len(content) > 1000 else content, height=150)
                    
                    if st.button("🔍 Classify Uploaded File", type="primary"):
                        with st.spinner("Processing file..."):
                            predictions = self.predict_email(content)
                            if predictions:
                                self.display_prediction_results(predictions, content)
                                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with tab3:
            self.display_model_performance()
        
        with tab4:
            st.markdown("## 🧪 Batch Email Testing")
            
            st.markdown("### Sample Email Tests")
            
            # Predefined test cases
            test_cases = {
                "Obvious Spam": "FREE! You've won $1,000,000! Click here NOW to claim your prize! Limited time offer!",
                "Phishing Attempt": "URGENT: Your account has been suspended. Click link to verify: http://fake-bank.com",
                "Promotion Email": "Get 50% off your next purchase! Use code SAVE50. Valid until midnight tonight!",
                "Legitimate Email": "Hi John, Thanks for the meeting yesterday. Please find the project report attached. Best regards, Sarah",
                "Work Email": "Team meeting moved to Thursday 3 PM in Conference Room B. Please update your calendars.",
                "Personal Email": "Hey! Are we still on for dinner tonight? Let me know what time works for you."
            }
            
            selected_test = st.selectbox("Choose a test email:", list(test_cases.keys()))
            
            if st.button("🧪 Test Selected Email"):
                test_email = test_cases[selected_test]
                st.markdown(f"**Testing:** {selected_test}")
                st.text_area("Email content:", test_email, height=100)
                
                with st.spinner("Testing..."):
                    predictions = self.predict_email(test_email)
                    if predictions:
                        self.display_prediction_results(predictions, test_email)

if __name__ == "__main__":
    app = ProfessionalSpamApp()
    app.run()
