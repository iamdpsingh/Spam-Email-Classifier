"""
Robust data processing module for spam email classification
Handles various CSV formats and ensures proper data cleaning
"""
import pandas as pd
import numpy as np
import re
import string
from textblob import TextBlob

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def load_and_parse_data(self, file_path):
        """
        Load and parse CSV data with robust format handling
        """
        print(f"📂 Loading data from: {file_path}")
        
        try:
            # Read CSV with various encodings
            for encoding in ['latin-1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, header=None)
                    print(f"✅ Successfully read with {encoding} encoding")
                    break
                except:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")
            
            print(f"📊 Raw data shape: {df.shape}")
            print(f"📋 Raw columns: {df.columns.tolist()}")
            
            # Parse the data based on format
            if df.shape[1] == 1:
                # Single column with tab separation (your format)
                print("🔍 Detected single-column tab-separated format")
                data = self._parse_single_column_tab_format(df)
            elif df.shape[1] >= 2:
                # Multiple columns
                print("🔍 Detected multi-column format")
                data = self._parse_multi_column_format(df)
            else:
                raise ValueError("Cannot parse data format")
            
            print(f"✅ Parsed data shape: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Creating sample dataset for demonstration")
            return self._create_sample_dataset()
    
    def _parse_single_column_tab_format(self, df):
        """Parse single column with tab-separated label and message"""
        labels = []
        messages = []
        
        for idx, row in df.iterrows():
            text = str(row.iloc[0])
            
            # Split by tab
            if '\t' in text:
                parts = text.split('\t', 1)
                if len(parts) >= 2:
                    label = parts[0].strip().lower()
                    message = parts[1].strip()
                    
                    # Only keep valid labels
                    if label in ['ham', 'spam']:
                        labels.append(label)
                        messages.append(message)
        
        return pd.DataFrame({
            'label': labels,
            'message': messages
        })
    
    def _parse_multi_column_format(self, df):
        """Parse multi-column format"""
        # Take first two columns
        df_clean = df.iloc[:, [0, 1]].copy()
        df_clean.columns = ['label', 'message']
        
        # Clean labels
        df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
        
        # Filter valid labels
        df_clean = df_clean[df_clean['label'].isin(['ham', 'spam'])]
        
        return df_clean
    
    def _create_sample_dataset(self):
        """Create high-quality sample dataset for testing"""
        spam_messages = [
            "WINNER!! You have won the £1,000,000 mega jackpot! Call 09061701461 to claim your prize NOW!",
            "URGENT! Your mobile number has been awarded £2000 cash prize! Text WIN to 80608 to claim!",
            "FREE entry to weekly competition! Text MUSIC to 80608. £1.50 per text. 18+ only",
            "Congratulations! You've been selected for a £1000 weekly prize draw. Call now!",
            "STOP! Your mobile has been selected for a £500 prize. Text CLAIM to 60300 now!",
            "Win £100 weekly by texting LOTTERY to 87121. £1 per text. Terms apply",
            "URGENT: Your credit card has been suspended. Call 0800123456 immediately!",
            "You have a new voicemail. Listen now by calling 09012345678. £1.50/min",
            "Claim your FREE £1000 shopping voucher! Visit website now limited time!",
            "FINAL NOTICE: Outstanding debt payment required. Call 0900876543 now!"
        ]
        
        ham_messages = [
            "Hey, are we still meeting for lunch tomorrow at the usual place?",
            "Thanks for sending the report. I'll review it and get back to you by Friday.",
            "The meeting has been rescheduled to 3 PM in conference room B.",
            "Can you pick up some milk on your way home? Thanks!",
            "Happy birthday! Hope you have a wonderful celebration with family.",
            "The project deadline has been extended by one week. Please update your schedules.",
            "Don't forget we have the team building event this Saturday at 10 AM.",
            "Your Amazon order has been dispatched and will arrive tomorrow.",
            "Could you send me the updated budget spreadsheet when you have a moment?",
            "Great presentation today! The client was very impressed with our proposal."
        ]
        
        return pd.DataFrame({
            'label': ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages),
            'message': spam_messages + ham_messages
        })
    
    def clean_and_preprocess_data(self, df):
        """
        Comprehensive data cleaning and preprocessing
        """
        print("🧹 Starting data cleaning...")
        
        # Remove null values
        initial_count = len(df)
        df = df.dropna()
        print(f"   Removed {initial_count - len(df)} null values")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"   Removed {initial_count - len(df)} duplicates")
        
        # Remove empty messages
        initial_count = len(df)
        df = df[df['message'].str.strip() != '']
        print(f"   Removed {initial_count - len(df)} empty messages")
        
        # Convert labels to binary
        df['target'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Verify label conversion
        if df['target'].isna().any():
            print("⚠️  Warning: Some labels could not be converted")
            df = df.dropna(subset=['target'])
        
        # Convert target to int
        df['target'] = df['target'].astype(int)
        
        print(f"✅ Final dataset: {len(df)} records")
        print(f"📊 Label distribution:")
        print(df['target'].value_counts())
        
        return df

    def basic_text_cleaning(self, text):
        """
        Basic text cleaning while preserving important spam indicators
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace but keep structure
        text = ' '.join(text.split())
        
        # Remove some special characters but keep important ones like £, $, !
        text = re.sub(r'[^\w\s£$!?.,()-]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
