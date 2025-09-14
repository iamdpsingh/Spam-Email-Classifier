"""
Robust feature extraction module for spam classification
Fixed NLTK compatibility and preprocessing issues
"""
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data"""
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with robust preprocessing"""
        self.stemmer = PorterStemmer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
        
        # Remove spam-important words from stopwords
        spam_words = {'free', 'win', 'winner', 'urgent', 'call', 'text', 'claim', 'prize', 'money'}
        self.stop_words = self.stop_words - spam_words
        
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
    
    def simple_tokenize(self, text):
        """Simple tokenization fallback"""
        # Basic word extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def advanced_text_preprocessing(self, text):
        """
        Robust text preprocessing with fallbacks
        """
        if pd.isna(text) or str(text).strip() == '':
            return 'empty message'
        
        try:
            text = str(text).lower()
            
            # Convert numbers to NUM token
            text = re.sub(r'\b\d+\b', 'NUM', text)
            
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s£$!]', ' ', text)
            
            # Try NLTK tokenization first
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple tokenization
                tokens = self.simple_tokenize(text)
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if len(token) > 2 and token not in self.stop_words:
                    try:
                        # Try stemming
                        stemmed = self.stemmer.stem(token)
                        filtered_tokens.append(stemmed)
                    except Exception:
                        # If stemming fails, use original token
                        filtered_tokens.append(token)
            
            # If no tokens remain, return basic cleaned text
            if not filtered_tokens:
                # Basic cleaning without NLTK
                words = re.findall(r'\b[a-zA-Z]+\b', text)
                filtered_tokens = [w for w in words if len(w) > 2]
            
            result = ' '.join(filtered_tokens) if filtered_tokens else 'empty'
            return result
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Fallback: basic text cleaning
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            words = text.split()
            words = [w for w in words if len(w) > 2]
            return ' '.join(words) if words else 'empty'
    
    def create_bow_features(self, texts, max_features=1000):
        """Create Bag of Words features with robust parameters"""
        print("🔤 Creating Bag of Words features...")
        
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),  # Only unigrams for robustness
            min_df=1,  # Very lenient
            max_df=0.99,  # Very lenient
            lowercase=True,
            strip_accents='unicode'
        )
        
        try:
            features = self.bow_vectorizer.fit_transform(texts)
            print(f"   BoW feature matrix shape: {features.shape}")
            return features
        except Exception as e:
            print(f"   Error creating BoW features: {e}")
            # Fallback: create minimal features
            return self._create_fallback_features(texts)
    
    def create_tfidf_features(self, texts, max_features=1000):
        """Create TF-IDF features with robust parameters"""
        print("📊 Creating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),  # Only unigrams for robustness
            min_df=1,  # Very lenient - minimum 1 document
            max_df=0.99,  # Very lenient
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True
        )
        
        try:
            features = self.tfidf_vectorizer.fit_transform(texts)
            print(f"   TF-IDF feature matrix shape: {features.shape}")
            return features
        except Exception as e:
            print(f"   Error creating TF-IDF features: {e}")
            print("   Falling back to simple BoW...")
            return self.create_bow_features(texts, max_features)
    
    def _create_fallback_features(self, texts):
        """Create minimal features as last resort"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Very basic vectorizer
        vectorizer = CountVectorizer(
            max_features=100,
            min_df=1,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]+\b'
        )
        
        return vectorizer.fit_transform(texts)
    
    def transform_bow(self, texts):
        """Transform new texts using fitted BoW vectorizer"""
        if self.bow_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted yet")
        return self.bow_vectorizer.transform(texts)
    
    def transform_tfidf(self, texts):
        """Transform new texts using fitted TF-IDF vectorizer"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted yet")
        return self.tfidf_vectorizer.transform(texts)
