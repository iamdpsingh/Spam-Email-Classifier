# 📧 Spam Email Classifier - Built for Your Dataset

A comprehensive machine learning solution that processes **your own email dataset** to build custom spam detection models using Naive Bayes and Logistic Regression algorithms.

## 🎯 Perfect for Your Dataset

This system is specifically designed to work with **your existing email data** in various formats and automatically clean, process, and train models from it.

## 🚀 Quick Start with Your Dataset

### Step 1: Setup
Create project folder
mkdir spam-email-classifier
cd spam-email-classifier

Copy all the provided code files into this directory
Install dependencies
pip install -r requirements.txt


### Step 2: Run with Your Dataset
Replace 'your_dataset.csv' with your actual dataset path
python run_pipeline.py --dataset path/to/your_dataset.csv --run-tests


### Step 3: Use the Web Application
The pipeline will automatically launch the web app, or run:
streamlit run streamlit_app.py


## 📊 Your Dataset Requirements

### ✅ Supported Formats
- **CSV files** (.csv) - Most common
- **Excel files** (.xlsx, .xls)
- **JSON files** (.json)
- **Text files** (.txt, .tsv)

### ✅ Required Content
Your dataset should have:
1. **Email text/content** (any column name - auto-detected)
2. **Labels** indicating spam/ham

### ✅ Label Formats (Auto-detected)
- `spam` / `ham`
- `1` / `0` (1=spam, 0=ham)
- `junk` / `legitimate`
- `bad` / `good`
- And many other variations!

### ✅ Example Dataset Formats

**CSV Format:**
email,label
"FREE! Win money now!",spam
"Hi, meeting at 2 PM tomorrow",ham
"URGENT account verification needed",spam


**Excel Format:**
| text | classification |
|------|----------------|
| FREE! Win money now! | spam |
| Hi, meeting at 2 PM tomorrow | ham |

**JSON Format:**
[
{"message": "FREE! Win money now!", "type": "spam"},
{"message": "Hi, meeting at 2 PM tomorrow", "type": "ham"}
]


## 🔧 System Features

### 🧹 **Universal Dataset Processor**
- **Auto-detects format** of your dataset file
- **Intelligent column detection** - finds text and label columns automatically
- **Smart label mapping** - handles various label formats
- **Data quality validation** - removes duplicates, handles missing data
- **Text preprocessing** - cleans and normalizes email content
- **Encoding detection** - handles different file encodings

### 🤖 **Machine Learning Pipeline**
- **Two algorithms**: Naive Bayes and Logistic Regression
- **TF-IDF feature extraction** with n-grams
- **Hyperparameter optimization** for best performance
- **Comprehensive evaluation** with multiple metrics
- **Model comparison** and automatic best model selection

### 🌐 **Interactive Web Application**
- **Real-time classification** of new emails
- **Batch processing** for multiple emails
- **File upload support** for easy testing
- **Performance visualizations** with interactive charts
- **Model comparison** interface
- **Professional UI** with custom styling

### 📊 **Advanced Analytics**
- **Confusion matrix analysis**
- **Performance metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Interactive visualizations** with Plotly
- **Model comparison charts**
- **Dataset quality analysis**

## 📁 Project Structure
spam-email-classifier/
├── data/
│ ├── raw/ # Your original dataset
│ └── processed/ # Cleaned dataset
├── src/
│ ├── data_preprocessing.py # Universal dataset cleaner
│ ├── model_training.py # ML model training
│ └── utils.py # Utility functions
├── models/ # Trained models (.pkl files)
├── results/ # Performance metrics
├── streamlit_app.py # Web application
├── run_pipeline.py # Main pipeline runner
├── config.py # Configuration settings
└── requirements.txt # Dependencies


## 🎯 Usage Examples

### Command Line Usage
Basic usage with your dataset
python run_pipeline.py --dataset my_emails.csv

Skip processing if data is already clean
python run_pipeline.py --dataset my_emails.csv --skip-processing

Train models only (skip web app launch)
python run_pipeline.py --dataset my_emails.csv --skip-streamlit

Run with testing
python run_pipeline.py --dataset my_emails.csv --run-tests


### Web Application Usage
1. **Dataset Processor**: Upload and clean your dataset
2. **Model Training**: Train ML models on your data
3. **Email Classifier**: Test spam detection in real-time
4. **Performance Analysis**: View detailed metrics and comparisons
5. **Visualizations**: Interactive charts and confusion matrices

## 📈 Expected Performance

### Typical Results with Good Datasets (500+ samples):
- **Accuracy**: 85-95%
- **Precision**: 90-100% (important for avoiding false positives)
- **Recall**: 75-90%
- **F1-Score**: 80-95%

### Performance depends on:
- **Dataset size** (more data = better performance)
- **Data quality** (clean, realistic emails)
- **Class balance** (similar amounts of spam and ham)
- **Label accuracy** (correctly labeled examples)

## 🔧 Advanced Configuration

Edit `config.py` to customize:
TF-IDF settings
TFIDF_CONFIG = {
'max_features': 5000, # Vocabulary size
'ngram_range': (1, 2), # Use unigrams and bigrams
'max_df': 0.9, # Remove very common words
'min_df': 1 # Minimum word frequency
}

Model parameters
NAIVE_BAYES_CONFIG = {
'alpha': 0.1 # Smoothing parameter
}

LOGISTIC_REGRESSION_CONFIG = {
'C': 10.0, # Regularization strength
'max_iter': 2000 # Maximum iterations
}


## 🧪 Testing Your Models

The system includes comprehensive testing:
Run automated tests
python run_pipeline.py --dataset your_data.csv --run-tests


Test samples include:
- Typical spam patterns
- Normal email communications  
- Edge cases and suspicious content

## 🚀 Deployment Options

### Local Development
streamlit run streamlit_app.py


### Cloud Deployment
1. **Streamlit Cloud**: Push to GitHub and deploy
2. **Heroku**: Use the provided configuration
3. **AWS/GCP**: Deploy using containers

## 📚 Troubleshooting

### Common Issues:

**"Dataset processing failed"**
- Check file format and encoding
- Ensure dataset has both text and label columns
- Verify file permissions

**"No models trained"**
- Ensure dataset processing completed successfully
- Check for sufficient data (minimum 10+ samples per class)
- Verify labels are in supported format

**Low performance**
- Increase dataset size (aim for 100+ samples per class)
- Improve data quality (remove duplicates, fix labels)
- Balance your dataset (similar spam/ham counts)

**Memory issues with large datasets**
- Process in smaller batches
- Reduce TF-IDF max_features in config.py
- Use more powerful hardware

## 📊 Dataset Quality Guidelines

### ✅ High-Quality Dataset Characteristics:
- **Size**: 200+ samples minimum, 1000+ preferred
- **Balance**: Similar amounts of spam and ham
- **Diversity**: Various types of spam and legitimate emails
- **Accuracy**: Correctly labeled examples
- **Realism**: Actual email content, not just keywords

### ❌ Common Dataset Problems:
- Too small (< 50 samples)
- Severely imbalanced (90% one class)
- Poor labeling accuracy
- Artificial or generated content
- Too many duplicates

## 🤝 Support & Contribution

### Getting Help:
1. Check the troubleshooting section
2. Review your dataset format
3. Ensure all dependencies are installed
4. Check the web application's error messages

### Contributing:
- Report bugs and issues
- Suggest improvements
- Add support for new file formats
- Enhance the machine learning pipeline

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🎉 Success Stories

This system works great with:
- **Personal email archives** (exported from Gmail, Outlook)
- **Corporate email datasets** (with privacy considerations)
- **Research datasets** (publicly available email collections)
- **Custom email collections** (scraped or collected data)

---

**🚀 Ready to build your custom spam classifier? Start with your dataset now!**

python run_pipeline.py --dataset path/to/your/emails.csv


**Your personalized spam detection system awaits!** 📧🤖
🏃‍♂️ How to Run Everything
Create the project folder and copy all files:

bash
mkdir spam-email-classifier
cd spam-email-classifier
# Copy all the code files above into respective locations
Install dependencies:

bash
pip install -r requirements.txt
Run with YOUR dataset:

bash
# Replace 'your_dataset.csv' with your actual dataset file path
python run_pipeline.py --dataset your_dataset.csv --run-tests
The system will:

✅ Automatically detect your dataset format

✅ Clean and process your data

✅ Train both ML models

✅ Test the models

✅ Launch the web application at http://localhost:8501

✅ To run this app - streamlit run streamlit_app.py
