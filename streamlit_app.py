import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import predict_email, load_model, validate_dataset, analyze_dataset, check_system_status
from src.data_preprocessing import UniversalDatasetCleaner, process_your_dataset
from config import MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}
.spam-alert {
    background-color: #ffebee;
    border: 2px solid #f44336;
    border-radius: 10px;
    padding: 1rem;
    color: #c62828;
}
.ham-safe {
    background-color: #e8f5e8;
    border: 2px solid #4caf50;
    border-radius: 10px;
    padding: 1rem;
    color: #2e7d32;
}
.status-good {
    color: #4caf50;
    font-weight: bold;
}
.status-warning {
    color: #ff9800;
    font-weight: bold;
}
.status-error {
    color: #f44336;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        model = load_model(model_path)
        if model:
            models[name] = model
    
    return models

@st.cache_data
def load_evaluation_data():
    """Load evaluation data"""
    comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    detailed_path = os.path.join(RESULTS_DIR, 'detailed_evaluation.csv')
    
    comparison_df = None
    detailed_df = None
    
    try:
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
        if os.path.exists(detailed_path):
            detailed_df = pd.read_csv(detailed_path)
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        
    return comparison_df, detailed_df

def predict_email_streamlit(text, model, model_name):
    """Streamlit-friendly email prediction"""
    if model is None:
        return None
        
    try:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probabilities[prediction] * 100
        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100
        
        return {
            'prediction': label,
            'confidence': confidence,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'model': model_name
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📧 Spam Email Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Powered by Machine Learning - Built for Your Dataset</p>', unsafe_allow_html=True)
    
    # Check system status
    system_ok = check_system_status()
    
    # Load models and data
    models = load_models()
    comparison_df, detailed_df = load_evaluation_data()
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # System status in sidebar
    st.sidebar.markdown("### 📊 System Status")
    if system_ok and models:
        st.sidebar.markdown('<p class="status-good">✅ System Ready</p>', unsafe_allow_html=True)
    elif models:
        st.sidebar.markdown('<p class="status-warning">⚠️ Partial Setup</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">❌ Setup Required</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown(f"**Models Available:** {len(models)}")
    
    # Model selection
    available_models = list(models.keys()) if models else ['No models available']
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        available_models,
        help="Select the machine learning model for prediction"
    )
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Home",
            "🔍 Email Classifier", 
            "📊 Dataset Processor",
            "🤖 Model Training",
            "📈 Performance Analysis",
            "📋 Visualizations",
            "ℹ️ About & Help"
        ]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(models, comparison_df, detailed_df)
    elif page == "🔍 Email Classifier":
        show_classifier_page(models, selected_model)
    elif page == "📊 Dataset Processor":
        show_dataset_processor_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "📈 Performance Analysis":
        show_performance_page(comparison_df, detailed_df)
    elif page == "📋 Visualizations":
        show_visualizations_page(detailed_df)
    elif page == "ℹ️ About & Help":
        show_about_page()

def show_home_page(models, comparison_df, detailed_df):
    """Home page with project overview"""
    st.markdown('<h2 class="sub-header">🎯 Your Custom Spam Email Classifier</h2>', unsafe_allow_html=True)
    
    # System status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Trained", len(models), f"{len(models)} Available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if comparison_df is not None and not comparison_df.empty:
            best_accuracy = comparison_df['Accuracy'].max()
            st.metric("Best Accuracy", f"{best_accuracy:.1%}", "Achieved")
        else:
            st.metric("Best Accuracy", "N/A", "Train models first")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if comparison_df is not None and not comparison_df.empty:
            best_precision = comparison_df['Precision'].max()
            st.metric("Best Precision", f"{best_precision:.1%}", "Achieved")
        else:
            st.metric("Best Precision", "N/A", "Train models first")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        dataset_status = "✅ Ready" if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'cleaned_dataset.csv')) else "⚠️ Process Dataset"
        st.metric("Dataset Status", dataset_status.split()[1], dataset_status.split()[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    if not models:
        st.warning("⚠️ No trained models found. Follow these steps to get started:")
        
        steps_col1, steps_col2 = st.columns(2)
        
        with steps_col1:
            st.markdown("""
            **Step 1: Prepare Your Dataset**
            1. Go to **📊 Dataset Processor**
            2. Upload your email dataset
            3. Let the system clean and process it
            
            **Step 2: Train Models**
            1. Go to **🤖 Model Training**
            2. Use your processed dataset
            3. Train Naive Bayes & Logistic Regression
            """)
        
        with steps_col2:
            st.markdown("""
            **Step 3: Test & Use**
            1. Go to **🔍 Email Classifier**
            2. Test with sample emails
            3. View performance metrics
            
            **Step 4: Analyze Results**
            1. Check **📈 Performance Analysis**
            2. View **📋 Visualizations**
            3. Compare model performance
            """)
    else:
        st.success("✅ System is ready! You can start classifying emails.")
        
        # Quick demo section
        st.markdown("### 🧪 Quick Demo")
        
        demo_samples = {
            "🚨 Typical Spam": "FREE! Win money now! Click here immediately for your prize!",
            "✅ Normal Email": "Hi, the quarterly meeting is scheduled for next Monday at 2 PM.",
            "⚠️ Suspicious": "URGENT! Your account needs immediate verification to avoid closure!"
        }
        
        selected_demo = st.selectbox("Choose a demo email:", list(demo_samples.keys()))
        demo_text = demo_samples[selected_demo]
        
        st.text_area("Demo email content:", demo_text, height=100, disabled=True)
        
        if st.button("🔍 Classify This Email", type="primary"):
            # Use the first available model for demo
            model_name = list(models.keys())[0]
            model = models[model_name]
            
            result = predict_email_streamlit(demo_text, model, model_name)
            
            if result:
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['prediction'] == 'SPAM':
                        st.markdown(f"""
                        <div class="spam-alert">
                            <h4>🚨 SPAM DETECTED</h4>
                            <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                            <p><strong>Model:</strong> {result["model"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ham-safe">
                            <h4>✅ LEGITIMATE EMAIL</h4>
                            <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                            <p><strong>Model:</strong> {result["model"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Quick probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['HAM', 'SPAM'],
                            y=[result['ham_probability'], result['spam_probability']],
                            marker_color=['#4CAF50', '#F44336']
                        )
                    ])
                    fig.update_layout(title="Probabilities", height=300)
                    st.plotly_chart(fig, use_container_width=True)

def show_classifier_page(models, selected_model):
    """Email classifier page"""
    st.markdown('<h2 class="sub-header">🔍 Email Spam Classifier</h2>', unsafe_allow_html=True)
    
    if selected_model == 'No models available':
        st.error("❌ No trained models found.")
        st.info("👆 Go to **🤖 Model Training** to train models with your dataset first.")
        return
    
    st.info(f"🤖 Using **{selected_model}** model for classification")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📝 Type/Paste Email", "📁 Upload Text File", "📋 Batch Processing"],
        horizontal=True
    )
    
    if input_method == "📝 Type/Paste Email":
        show_text_input_classifier(models, selected_model)
    elif input_method == "📁 Upload Text File":
        show_file_upload_classifier(models, selected_model)
    elif input_method == "📋 Batch Processing":
        show_batch_processing(models, selected_model)

def show_text_input_classifier(models, selected_model):
    """Text input classifier interface"""
    email_text = st.text_area(
        "📧 Enter email content:",
        height=200,
        placeholder="Paste your email content here...\n\nExample:\nSubject: Meeting Tomorrow\nHi team, just a reminder that we have our weekly meeting tomorrow at 2 PM.",
        help="Enter the complete email text including subject and body"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        classify_button = st.button("🔍 Classify Email", type="primary", use_container_width=True)
    
    if classify_button and email_text.strip():
        model = models.get(selected_model)
        if model:
            with st.spinner("Analyzing email..."):
                result = predict_email_streamlit(email_text, model, selected_model)
            
            if result:
                # Display results
                st.markdown("### 📊 Classification Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if result['prediction'] == 'SPAM':
                        st.markdown(f"""
                        <div class="spam-alert">
                            <h3>🚨 SPAM DETECTED</h3>
                            <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                            <p><strong>Model Used:</strong> {result["model"]}</p>
                            <p><strong>Recommendation:</strong> This email should be filtered or carefully reviewed</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ham-safe">
                            <h3>✅ LEGITIMATE EMAIL</h3>
                            <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                            <p><strong>Model Used:</strong> {result["model"]}</p>
                            <p><strong>Recommendation:</strong> This email appears safe to read and respond to</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Detailed probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['HAM (Legitimate)', 'SPAM'],
                            y=[result['ham_probability'], result['spam_probability']],
                            marker_color=['#4CAF50', '#F44336'],
                            text=[f'{result["ham_probability"]:.1f}%', f'{result["spam_probability"]:.1f}%'],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="📈 Classification Probabilities",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.markdown("### 🔍 Analysis Insights")
                
                insights = []
                if result['spam_probability'] > 80:
                    insights.append("⚠️ Very high spam probability - likely contains spam indicators")
                elif result['spam_probability'] > 50:
                    insights.append("🔍 Moderate spam probability - review content carefully")
                else:
                    insights.append("✅ Low spam probability - appears to be legitimate")
                
                if result['confidence'] > 90:
                    insights.append("🎯 High confidence prediction - model is very certain")
                elif result['confidence'] > 70:
                    insights.append("📊 Good confidence level - reliable prediction")
                else:
                    insights.append("⚠️ Lower confidence - consider additional verification")
                
                for insight in insights:
                    st.write(insight)
        else:
            st.error("Selected model not available")

def show_file_upload_classifier(models, selected_model):
    """File upload classifier interface"""
    uploaded_file = st.file_uploader(
        "📁 Upload email file:",
        type=['txt', 'csv'],
        help="Upload a text file containing email content or CSV with emails"
    )
    
    if uploaded_file is not None:
        model = models.get(selected_model)
        if model:
            try:
                if uploaded_file.type == "text/plain":
                    # Handle text file
                    email_content = str(uploaded_file.read(), "utf-8")
                    
                    st.success("📧 File loaded successfully!")
                    st.text_area("Email content:", email_content[:500] + "..." if len(email_content) > 500 else email_content, height=200)
                    
                    if st.button("🔍 Classify This Email"):
                        with st.spinner("Analyzing email..."):
                            result = predict_email_streamlit(email_content, model, selected_model)
                        
                        if result:
                            if result['prediction'] == 'SPAM':
                                st.error(f"🚨 SPAM DETECTED (Confidence: {result['confidence']:.1f}%)")
                            else:
                                st.success(f"✅ LEGITIMATE EMAIL (Confidence: {result['confidence']:.1f}%)")
                
                elif uploaded_file.type == "application/vnd.ms-excel" or uploaded_file.name.endswith('.csv'):
                    # Handle CSV file
                    df = pd.read_csv(uploaded_file)
                    st.success(f"📊 CSV file loaded with {len(df)} rows!")
                    
                    st.dataframe(df.head())
                    
                    # Let user select text column
                    text_column = st.selectbox("Select text column:", df.columns)
                    
                    if st.button("🔍 Classify All Emails"):
                        with st.spinner(f"Processing {len(df)} emails..."):
                            results = []
                            
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(df[text_column]):
                                result = predict_email_streamlit(str(text), model, selected_model)
                                if result:
                                    results.append({
                                        'Email': str(text)[:50] + "..." if len(str(text)) > 50 else str(text),
                                        'Prediction': result['prediction'],
                                        'Confidence': f"{result['confidence']:.1f}%"
                                    })
                                
                                progress_bar.progress((i + 1) / len(df))
                            
                            # Display results
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                            
                            # Summary
                            spam_count = sum(1 for r in results if r['Prediction'] == 'SPAM')
                            ham_count = len(results) - spam_count
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Emails", len(results))
                            with col2:
                                st.metric("SPAM Detected", spam_count)
                            with col3:
                                st.metric("HAM (Legitimate)", ham_count)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.error("Selected model not available")

def show_batch_processing(models, selected_model):
    """Batch processing interface"""
    st.info("📝 Enter multiple emails separated by '---' for batch processing")
    
    batch_text = st.text_area(
        "Enter multiple emails:",
        height=300,
        placeholder="Email 1 content here...\n---\nEmail 2 content here...\n---\nEmail 3 content here...",
        help="Separate each email with '---' on a new line"
    )
    
    if st.button("🔄 Process Batch", type="primary") and batch_text.strip():
        model = models.get(selected_model)
        if model:
            emails = [email.strip() for email in batch_text.split('---') if email.strip()]
            
            if emails:
                with st.spinner(f"Processing {len(emails)} emails..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, email in enumerate(emails):
                        result = predict_email_streamlit(email, model, selected_model)
                        if result:
                            results.append({
                                'Email #': i + 1,
                                'Preview': email[:50] + '...' if len(email) > 50 else email,
                                'Prediction': result['prediction'],
                                'Confidence': f"{result['confidence']:.1f}%"
                            })
                        
                        progress_bar.progress((i + 1) / len(emails))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                spam_count = sum(1 for r in results if r['Prediction'] == 'SPAM')
                ham_count = len(results) - spam_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Emails", len(results))
                with col2:
                    st.metric("SPAM Detected", spam_count, f"{spam_count/len(results)*100:.1f}%")
                with col3:
                    st.metric("HAM (Legitimate)", ham_count, f"{ham_count/len(results)*100:.1f}%")
                
                # Download results
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv_data,
                    file_name="batch_classification_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid emails found. Make sure to separate emails with '---'")
        else:
            st.error("Selected model not available")

def show_dataset_processor_page():
    """Dataset processing and cleaning page"""
    st.markdown('<h2 class="sub-header">📊 Dataset Processor</h2>', unsafe_allow_html=True)
    st.markdown("Upload and process your email dataset for training spam classification models.")
    
    # Upload section
    st.markdown("### 📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose your dataset file:",
        type=['csv', 'xlsx', 'xls', 'json', 'txt', 'tsv'],
        help="Support formats: CSV, Excel, JSON, TXT, TSV"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Process dataset button
        if st.button("🧹 Process Dataset", type="primary"):
            with st.spinner("Processing your dataset... This may take a moment."):
                # Process the dataset
                output_path = process_your_dataset(upload_path, PROCESSED_DATA_DIR)
                
                if output_path:
                    st.success("✅ Dataset processed successfully!")
                    
                    # Load and display processed dataset
                    processed_df = pd.read_csv(output_path)
                    
                    st.markdown("### 📊 Processed Dataset Preview")
                    st.dataframe(processed_df.head(10))
                    
                    # Dataset statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Samples", len(processed_df))
                    
                    with col2:
                        if 'label' in processed_df.columns:
                            spam_count = (processed_df['label'] == 'spam').sum()
                            st.metric("Spam Emails", spam_count)
                    
                    with col3:
                        if 'label' in processed_df.columns:
                            ham_count = (processed_df['label'] == 'ham').sum()
                            st.metric("Ham Emails", ham_count)
                    
                    with col4:
                        if 'text' in processed_df.columns:
                            avg_length = processed_df['text'].str.len().mean()
                            st.metric("Avg Text Length", f"{avg_length:.0f} chars")
                    
                    # Show label distribution
                    if 'label' in processed_df.columns:
                        st.markdown("### 📈 Label Distribution")
                        label_counts = processed_df['label'].value_counts()
                        
                        fig = go.Figure(data=[
                            go.Bar(x=label_counts.index, y=label_counts.values, 
                                  marker_color=['#4CAF50', '#F44336'])
                        ])
                        fig.update_layout(title="Email Label Distribution", 
                                        xaxis_title="Label", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download processed dataset
                    csv_data = processed_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Processed Dataset",
                        data=csv_data,
                        file_name="processed_dataset.csv",
                        mime="text/csv"
                    )
                    
                    st.info("🚀 Your dataset is ready! Go to **🤖 Model Training** to train your models.")
                
                else:
                    st.error("❌ Dataset processing failed. Please check your file format and try again.")
    
    # Show existing processed datasets
    st.markdown("### 📋 Existing Processed Datasets")
    
    processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]
    
    if processed_files:
        selected_file = st.selectbox("Select processed dataset:", processed_files)
        
        if selected_file:
            file_path = os.path.join(PROCESSED_DATA_DIR, selected_file)
            df = pd.read_csv(file_path)
            
            st.write(f"**Dataset:** {selected_file}")
            st.write(f"**Size:** {df.shape[0]} samples, {df.shape[1]} columns")
            
            if st.checkbox("Show dataset preview"):
                st.dataframe(df.head())
            
            # Analyze button
            if st.button("📊 Analyze Dataset"):
                analyze_dataset(df)
    else:
        st.info("No processed datasets found. Upload and process a dataset first.")

def show_model_training_page():
    """Model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    st.markdown("Train machine learning models on your processed dataset.")
    
    # Check for processed datasets
    processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]
    
    if not processed_files:
        st.warning("⚠️ No processed datasets found.")
        st.info("👆 Go to **📊 Dataset Processor** to upload and process your dataset first.")
        return
    
    # Dataset selection
    st.markdown("### 📊 Select Dataset")
    selected_dataset = st.selectbox("Choose dataset for training:", processed_files)
    
    if selected_dataset:
        dataset_path = os.path.join(PROCESSED_DATA_DIR, selected_dataset)
        
        # Load and validate dataset
        try:
            df = pd.read_csv(dataset_path)
            st.success(f"✅ Dataset loaded: {df.shape[0]} samples")
            
            # Show dataset info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            
            with col2:
                if 'label' in df.columns:
                    spam_count = (df['label'] == 'spam').sum()
                    st.metric("Spam Emails", spam_count)
            
            with col3:
                if 'label' in df.columns:
                    ham_count = (df['label'] == 'ham').sum()
                    st.metric("Ham Emails", ham_count)
            
            # Validate dataset
            if validate_dataset(df):
                st.success("✅ Dataset validation passed!")
                
                # Training configuration
                st.markdown("### ⚙️ Training Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05, 
                                        help="Portion of data to use for testing")
                
                with col2:
                    random_state = st.number_input("Random State", value=42, 
                                                 help="For reproducible results")
                
                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    st.markdown("**TF-IDF Parameters:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        max_features = st.number_input("Max Features", value=5000, min_value=1000)
                        min_df = st.number_input("Min Document Frequency", value=1, min_value=1)
                    
                    with col2:
                        max_df = st.slider("Max Document Frequency", 0.1, 1.0, 0.9, 0.1)
                        ngram_range = st.selectbox("N-gram Range", ["(1,1)", "(1,2)", "(1,3)"], index=1)
                
                # Training button
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        # Import training function
                        from src.model_training import train_spam_classifiers
                        
                        # Train models
                        try:
                            results = train_spam_classifiers(dataset_path)
                            
                            if results:
                                classifier, comparison_df, detailed_df, best_model = results
                                
                                st.success("🎉 Training completed successfully!")
                                
                                # Show results
                                st.markdown("### 📊 Training Results")
                                
                                if comparison_df is not None:
                                    st.dataframe(comparison_df)
                                    
                                    st.success(f"🏆 Best Model: **{best_model}**")
                                    
                                    # Performance metrics
                                    best_row = comparison_df[comparison_df['Model'] == best_model].iloc[0]
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Accuracy", f"{best_row['Accuracy']:.1%}")
                                    with col2:
                                        st.metric("Precision", f"{best_row['Precision']:.1%}")
                                    with col3:
                                        st.metric("Recall", f"{best_row['Recall']:.1%}")
                                    with col4:
                                        st.metric("F1-Score", f"{best_row['F1-Score']:.1%}")
                                
                                st.info("🎯 Your models are now ready! Go to **🔍 Email Classifier** to test them.")
                            
                            else:
                                st.error("❌ Training failed. Please check your dataset and try again.")
                        
                        except Exception as e:
                            st.error(f"❌ Training error: {str(e)}")
            
            else:
                st.error("❌ Dataset validation failed. Please check your dataset format.")
        
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

def show_performance_page(comparison_df, detailed_df):
    """Model performance analysis page"""
    st.markdown('<h2 class="sub-header">📈 Performance Analysis</h2>', unsafe_allow_html=True)
    
    if comparison_df is not None and not comparison_df.empty:
        # Model comparison
        st.markdown("### 🏆 Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model highlight
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df['F1-Score'].max()
        
        st.success(f"🥇 **Best Performing Model:** {best_model} (F1-Score: {best_f1:.4f})")
        
        # Performance visualization
        st.markdown("### 📊 Performance Metrics Comparison")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for model in comparison_df['Model']:
            model_data = comparison_df[comparison_df['Model'] == model].iloc[0]
            values = [model_data[metric] for metric in metrics if metric in comparison_df.columns]
            available_metrics = [metric for metric in metrics if metric in comparison_df.columns]
            
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        if detailed_df is not None and not detailed_df.empty:
            st.markdown("### 📋 Detailed Metrics")
            
            with st.expander("Show detailed metrics table"):
                st.dataframe(detailed_df, use_container_width=True)
        
        # Performance insights
        st.markdown("### 🔍 Performance Insights")
        
        insights = []
        
        # Check for perfect precision
        if (comparison_df['Precision'] == 1.0).any():
            perfect_models = comparison_df[comparison_df['Precision'] == 1.0]['Model'].tolist()
            insights.append(f"🎯 **Perfect Precision**: {', '.join(perfect_models)} achieved 100% precision (no false positives)")
        
        # Check accuracy
        best_accuracy = comparison_df['Accuracy'].max()
        if best_accuracy > 0.9:
            insights.append(f"🎯 **Excellent Accuracy**: Best model achieved {best_accuracy:.1%} accuracy")
        elif best_accuracy > 0.8:
            insights.append(f"📊 **Good Accuracy**: Best model achieved {best_accuracy:.1%} accuracy")
        
        # Check F1-Score
        if best_f1 > 0.8:
            insights.append(f"⭐ **High F1-Score**: Excellent balance between precision and recall ({best_f1:.3f})")
        
        for insight in insights:
            st.write(insight)
        
        # Metrics explanation
        with st.expander("📖 Understanding the Metrics"):
            st.markdown("""
            **Accuracy**: Overall correctness of predictions (correct predictions / total predictions)
            
            **Precision**: Of all emails predicted as spam, how many were actually spam? (crucial for avoiding false positives)
            
            **Recall**: Of all actual spam emails, how many were correctly identified? (spam detection rate)
            
            **F1-Score**: Harmonic mean of precision and recall (balanced measure)
            
            **ROC-AUC**: Area under the ROC curve (model's ability to distinguish between classes)
            
            **For spam detection, high precision is often more important than high recall to avoid blocking legitimate emails.**
            """)
    
    else:
        st.warning("⚠️ No performance data available.")
        st.info("👆 Go to **🤖 Model Training** to train models and generate performance metrics.")

def show_visualizations_page(detailed_df):
    """Visualizations page"""
    st.markdown('<h2 class="sub-header">📋 Performance Visualizations</h2>', unsafe_allow_html=True)
    
    if detailed_df is not None and not detailed_df.empty:
        # Confusion matrix visualization
        st.markdown("### 🎯 Confusion Matrix Analysis")
        
        for idx, row in detailed_df.iterrows():
            model_name = row['Model']
            
            # Create confusion matrix data
            conf_matrix_data = np.array([
                [row['True_Negatives'], row['False_Positives']],
                [row['False_Negatives'], row['True_Positives']]
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {model_name}")
                
                fig = px.imshow(
                    conf_matrix_data,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Ham', 'Spam'],
                    y=['Ham', 'Spam'],
                    color_continuous_scale='Blues',
                    aspect="auto"
                )
                
                # Add text annotations
                for i in range(len(conf_matrix_data)):
                    for j in range(len(conf_matrix_data[0])):
                        fig.add_annotation(
                            x=j, y=i,
                            text=str(int(conf_matrix_data[i][j])),
                            showarrow=False,
                            font=dict(color="white" if conf_matrix_data[i][j] > conf_matrix_data.max()/2 else "black", size=16)
                        )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance metrics for this model
                st.markdown("#### Key Metrics")
                
                st.metric("True Positives", int(row['True_Positives']), "Spam correctly identified")
                st.metric("True Negatives", int(row['True_Negatives']), "Ham correctly identified")
                st.metric("False Positives", int(row['False_Positives']), "Ham incorrectly marked as spam")
                st.metric("False Negatives", int(row['False_Negatives']), "Spam missed")
                
                # Additional metrics
                st.metric("Specificity", f"{row['Specificity']:.3f}", "True negative rate")
                if 'NPV' in row:
                    st.metric("NPV", f"{row['NPV']:.3f}", "Negative predictive value")
        
        # Model comparison radar chart
        st.markdown("### 📊 Model Performance Radar")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
        available_metrics = [m for m in metrics if m in detailed_df.columns]
        
        fig = go.Figure()
        
        for idx, row in detailed_df.iterrows():
            values = [row[metric] for metric in available_metrics]
            values.append(values[0])  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Comparison (Radar Chart)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No visualization data available.")
        st.info("👆 Train models first to generate performance visualizations.")

def show_about_page():
    """About and help page"""
    st.markdown('<h2 class="sub-header">ℹ️ About & Help</h2>', unsafe_allow_html=True)
    
    # Project overview
    st.markdown("### 🎯 About This Project")
    st.markdown("""
    This **Spam Email Classifier** is a comprehensive machine learning solution designed to work with **your own dataset**. 
    It provides end-to-end functionality from data processing to model deployment.
    
    #### 🌟 Key Features:
    - **Universal Dataset Processor**: Handles various formats (CSV, Excel, JSON, TXT)
    - **Automatic Data Cleaning**: Intelligent preprocessing and validation
    - **Two ML Algorithms**: Naive Bayes and Logistic Regression comparison
    - **Real-time Classification**: Instant spam detection with confidence scores
    - **Interactive Visualizations**: Comprehensive performance analysis
    - **Professional Interface**: User-friendly web application
    """)
    
    # How to use guide
    st.markdown("### 📚 How to Use")
    
    with st.expander("🚀 Getting Started (First Time)"):
        st.markdown("""
        1. **📊 Upload Your Dataset**: Go to "Dataset Processor" and upload your email data
        2. **🧹 Clean & Process**: Let the system automatically clean and standardize your data
        3. **🤖 Train Models**: Go to "Model Training" to train Naive Bayes and Logistic Regression
        4. **🔍 Test Classification**: Use "Email Classifier" to test your models
        5. **📈 Analyze Results**: Review performance in "Performance Analysis"
        """)
    
    with st.expander("📊 Dataset Requirements"):
        st.markdown("""
        **Your dataset should contain:**
        - **Email content/text** (in any column - will be auto-detected)
        - **Labels** indicating spam/ham (various formats accepted)
        
        **Supported formats:**
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - JSON files (.json)
        - Text files (.txt, .tsv)
        
        **Supported label formats:**
        - spam/ham
        - 1/0 (1=spam, 0=ham)
        - Various other formats (auto-detected)
        """)
    
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues & Solutions:**
            
            **Problem**: "No models available"
            - **Solution**: Go to "Model Training" and train models first
            
            **Problem**: "Dataset processing failed"
            - **Solution**: Check your dataset format - ensure it has text and label columns
            
            **Problem**: "Model training failed"
            - **Solution**: Ensure your dataset is processed correctly and has both spam and ham examples
            
            **Problem**: Low model performance
            - **Solution**: Check dataset quality, balance, and size (minimum 50+ samples recommended)
            
            **Problem**: File upload issues
            - **Solution**: Ensure file format is supported and file size is reasonable
        """)
    
    with st.expander("📈 Improving Performance"):
        st.markdown("""
        **Tips for Better Results:**
        
        1. **Dataset Quality**:
           - Use at least 100+ samples for each class (spam/ham)
           - Ensure realistic email content (not just keywords)
           - Balance your dataset (similar amounts of spam and ham)
        
        2. **Data Preprocessing**:
           - Remove duplicates and low-quality samples
           - Ensure consistent labeling
           - Include diverse spam and legitimate email types
        
        3. **Model Selection**:
           - Naive Bayes: Better for smaller datasets, faster training
           - Logistic Regression: Often better performance with larger datasets
        
        4. **Performance Metrics**:
           - For email filtering, prioritize **Precision** to avoid false positives
           - **F1-Score** gives balanced view of overall performance
           - **Accuracy** can be misleading with imbalanced datasets
        """)
    
    # Technical details
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - Algorithms: Multinomial Naive Bayes, Logistic Regression
        - Feature Extraction: TF-IDF with n-grams
        - Text Preprocessing: Cleaning, normalization, stop word removal
        - Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
        """)
    
    with col2:
        st.markdown("""
        **Technology Stack:**
        - Python 3.8+
        - Scikit-learn (Machine Learning)
        - Streamlit (Web Interface)
        - Plotly (Visualizations)
        - Pandas (Data Processing)
        - NumPy (Numerical Computing)
        """)
    
    # Contact and support
    st.markdown("### 📞 Support & Contact")
    
    st.markdown("""
    **Need Help?**
    - Check the troubleshooting section above
    - Review your dataset format and quality
    - Ensure all steps are followed in order
    
    **Project Information:**
    - Built for educational and research purposes
    - Suitable for small to medium-scale email filtering
    - Extensible architecture for custom enhancements
    
    **Performance Notes:**
    - Processing time depends on dataset size
    - Larger datasets (1000+ samples) may take several minutes to process
    - Models are saved automatically and persist between sessions
    """)
    
    # System status
    st.markdown("### 🖥️ Current System Status")
    
    models = load_models()
    processed_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.csv')]
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if models:
            st.success(f"✅ {len(models)} Models Ready")
        else:
            st.warning("⚠️ No Models Trained")
    
    with status_col2:
        if processed_files:
            st.success(f"✅ {len(processed_files)} Datasets Processed")
        else:
            st.warning("⚠️ No Processed Datasets")
    
    with status_col3:
        if models and processed_files:
            st.success("✅ System Ready")
        else:
            st.info("ℹ️ Setup Required")

if __name__ == "__main__":
    main()

