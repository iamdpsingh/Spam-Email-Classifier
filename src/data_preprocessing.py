import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
import os
import chardet
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EmailPreprocessor:
    """
    Advanced email preprocessing class for spam detection
    """
    
    def __init__(self):
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        ])
    
    def clean_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words and short words
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_dataset(self, df, text_column='text', label_column='label'):
        """Preprocess entire dataset"""
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        processed_df = processed_df[processed_df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
        
        # Encode labels
        if processed_df[label_column].dtype == 'object':
            label_mapping = self._create_label_mapping(processed_df[label_column])
            processed_df['label_encoded'] = processed_df[label_column].map(label_mapping)
        else:
            processed_df['label_encoded'] = processed_df[label_column]
        
        return processed_df
    
    def _create_label_mapping(self, labels):
        """Create label mapping for various label formats"""
        unique_labels = labels.unique()
        
        # Common spam indicators
        spam_indicators = ['spam', 'junk', '1', 1, 'unsolicited', 'unwanted', 'bad', 'malicious']
        ham_indicators = ['ham', 'legitimate', '0', 0, 'good', 'normal', 'safe', 'clean']
        
        label_mapping = {}
        
        for label in unique_labels:
            label_str = str(label).lower().strip()
            
            if any(spam_ind in label_str for spam_ind in spam_indicators):
                label_mapping[label] = 1
            elif any(ham_ind in label_str for ham_ind in ham_indicators):
                label_mapping[label] = 0
            else:
                # Try numeric conversion
                try:
                    numeric_val = float(label_str)
                    label_mapping[label] = 1 if numeric_val > 0.5 else 0
                except:
                    label_mapping[label] = 0  # Default to ham if uncertain
        
        return label_mapping


class UniversalDatasetCleaner:
    """
    Universal dataset cleaning pipeline that handles various formats and quality issues
    """
    
    def __init__(self):
        self.cleaned_data = None
        self.cleaning_stats = {}
        self.detected_encoding = None
        self.detected_format = None
    
    def detect_encoding(self, file_path):
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                print(f"🔍 Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
        except Exception as e:
            print(f"⚠️ Encoding detection failed: {e}")
            return 'utf-8'  # Default fallback
    
    def detect_format_and_clean(self, file_path, output_path=None):
        """Automatically detect dataset format and clean it"""
        file_ext = Path(file_path).suffix.lower()
        self.detected_format = file_ext
        
        print(f"🔍 Processing file: {file_path}")
        print(f"🔍 Detected format: {file_ext}")
        
        try:
            # Detect encoding first
            if file_ext in ['.csv', '.txt']:
                self.detected_encoding = self.detect_encoding(file_path)
            
            # Load dataset based on file extension
            if file_ext == '.csv':
                df = self._load_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            elif file_ext == '.json':
                df = self._load_json(file_path)
            elif file_ext == '.txt':
                df = self._load_text(file_path)
            elif file_ext == '.tsv':
                df = self._load_tsv(file_path)
            else:
                # Try to guess format
                df = self._try_multiple_formats(file_path)
                if df is None:
                    raise ValueError(f"Unsupported or unreadable file format: {file_ext}")
            
            print(f"✅ Successfully loaded dataset: {df.shape}")
            print(f"📊 Columns found: {list(df.columns)}")
            print(f"📋 Data types: {df.dtypes.to_dict()}")
            
            # Clean the dataset
            cleaned_df = self._universal_cleaning_pipeline(df)
            
            # Save cleaned dataset if output path provided
            if output_path:
                self._save_cleaned_dataset(cleaned_df, output_path)
            
            self.cleaned_data = cleaned_df
            return cleaned_df
            
        except Exception as e:
            print(f"❌ Error processing dataset: {str(e)}")
            return None
    
    def _load_csv(self, file_path):
        """Load CSV file with multiple fallback options"""
        encodings_to_try = [self.detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators_to_try = [',', ';', '\t', '|']
        
        for encoding in encodings_to_try:
            for sep in separators_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
                    if len(df.columns) >= 2:  # Need at least 2 columns
                        print(f"✅ CSV loaded with encoding: {encoding}, separator: '{sep}'")
                        return df
                except Exception as e:
                    continue
        
        raise ValueError("Could not load CSV file with any encoding/separator combination")
    
    def _load_excel(self, file_path):
        """Load Excel file"""
        try:
            # Try to load all sheets and find the best one
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    if len(df) > 0 and len(df.columns) >= 2:
                        print(f"✅ Excel loaded from sheet: {sheet_name}")
                        return df
                except Exception as e:
                    continue
            
            # If no good sheet found, load the first one
            df = pd.read_excel(file_path)
            return df
            
        except Exception as e:
            raise ValueError(f"Could not load Excel file: {e}")
    
    def _load_json(self, file_path):
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding=self.detected_encoding or 'utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not load JSON file: {e}")
    
    def _load_text(self, file_path):
        """Load plain text file"""
        try:
            # Try different approaches for text files
            with open(file_path, 'r', encoding=self.detected_encoding or 'utf-8') as f:
                lines = f.readlines()
            
            # Option 1: Tab-separated values
            if '\t' in lines[0]:
                return pd.read_csv(file_path, sep='\t', encoding=self.detected_encoding or 'utf-8')
            
            # Option 2: Space-separated values
            elif ' ' in lines[0]:
                return pd.read_csv(file_path, sep=' ', encoding=self.detected_encoding or 'utf-8')
            
            # Option 3: One email per line format
            else:
                emails = [line.strip() for line in lines if line.strip()]
                return pd.DataFrame({'text': emails, 'label': 'unknown'})
                
        except Exception as e:
            raise ValueError(f"Could not load text file: {e}")
    
    def _load_tsv(self, file_path):
        """Load TSV (Tab-separated values) file"""
        return pd.read_csv(file_path, sep='\t', encoding=self.detected_encoding or 'utf-8')
    
    def _try_multiple_formats(self, file_path):
        """Try multiple formats if extension is unknown"""
        loaders = [
            ('CSV', self._load_csv),
            ('TSV', self._load_tsv),
            ('JSON', self._load_json),
            ('Text', self._load_text)
        ]
        
        for format_name, loader in loaders:
            try:
                df = loader(file_path)
                print(f"✅ Successfully loaded as {format_name}")
                return df
            except Exception as e:
                continue
        
        return None
    
    def _universal_cleaning_pipeline(self, df):
        """Universal cleaning pipeline"""
        print("\n🧹 Starting universal cleaning pipeline...")
        
        # Store original stats
        original_rows = len(df)
        self.cleaning_stats['original_rows'] = original_rows
        
        # Step 1: Handle different column naming conventions
        df = self._standardize_columns(df)
        
        # Step 2: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 3: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 4: Clean text content
        df = self._clean_text_content(df)
        
        # Step 5: Standardize labels
        df = self._standardize_labels(df)
        
        # Step 6: Remove invalid entries
        df = self._remove_invalid_entries(df)
        
        # Step 7: Final validation
        df = self._final_validation(df)
        
        # Update stats
        self.cleaning_stats['final_rows'] = len(df)
        self.cleaning_stats['rows_removed'] = original_rows - len(df)
        self.cleaning_stats['removal_percentage'] = (self.cleaning_stats['rows_removed'] / original_rows) * 100
        
        self._print_cleaning_summary()
        
        return df
    
    def _standardize_columns(self, df):
        """Standardize column names regardless of original format"""
        print("📝 Standardizing column names...")
        
        # Print original columns for debugging
        print(f"   Original columns: {list(df.columns)}")
        
        # Common column name variations for text content
        text_columns = [
            'message', 'text', 'email', 'content', 'body', 'mail', 'sms', 'msg',
            'email_text', 'message_text', 'content_text', 'email_body', 'message_body',
            'v2', 'column2', 'col2', 'field2', 'data', 'description'
        ]
        
        # Common column name variations for labels
        label_columns = [
            'label', 'class', 'category', 'type', 'spam', 'classification', 'target',
            'is_spam', 'spam_label', 'class_label', 'y', 'output', 'result',
            'v1', 'column1', 'col1', 'field1', 'status'
        ]
        
        # Find text column
        text_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in text_columns:
                text_col = col
                break
            # Check if column name contains text-related keywords
            if any(keyword in col_lower for keyword in ['message', 'text', 'email', 'content', 'body']):
                text_col = col
                break
        
        # Find label column
        label_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in label_columns:
                label_col = col
                break
            # Check if column name contains label-related keywords
            if any(keyword in col_lower for keyword in ['label', 'class', 'spam', 'category', 'target']):
                label_col = col
                break
        
        # If specific columns not found, use heuristics
        if text_col is None or label_col is None:
            print("   Using heuristic column detection...")
            
            # Analyze columns by content
            for col in df.columns:
                sample_data = df[col].dropna().astype(str)
                if len(sample_data) > 0:
                    avg_length = sample_data.str.len().mean()
                    
                    # Text columns typically have longer content
                    if avg_length > 50 and text_col is None:
                        text_col = col
                    
                    # Label columns have short, categorical content
                    elif avg_length < 20 and len(sample_data.unique()) <= 10 and label_col is None:
                        label_col = col
        
        # Final fallback: use first two columns
        if text_col is None:
            if len(df.columns) > 1:
                # Use the column with longer average content as text
                col_lengths = {}
                for col in df.columns:
                    try:
                        avg_len = df[col].astype(str).str.len().mean()
                        col_lengths[col] = avg_len
                    except:
                        col_lengths[col] = 0
                
                text_col = max(col_lengths.keys(), key=lambda x: col_lengths[x])
            else:
                text_col = df.columns[0]
        
        if label_col is None:
            remaining_cols = [col for col in df.columns if col != text_col]
            if remaining_cols:
                label_col = remaining_cols[0]
            else:
                # If only one column, create a default label
                df['default_label'] = 'unknown'
                label_col = 'default_label'
        
        # Create standardized dataset
        standardized_df = pd.DataFrame({
            'text': df[text_col],
            'label': df[label_col]
        })
        
        print(f"   Text column: '{text_col}' -> 'text'")
        print(f"   Label column: '{label_col}' -> 'label'")
        
        return standardized_df
    
    def _remove_duplicates(self, df):
        """Remove duplicate entries"""
        print("🔄 Removing duplicates...")
        original_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Remove duplicates based on text content only (case-insensitive)
        df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        
        removed_count = original_count - len(df)
        self.cleaning_stats['duplicates_removed'] = removed_count
        print(f"   Removed {removed_count} duplicate entries")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in text and labels"""
        print("🚫 Handling missing values...")
        
        # Count missing values
        missing_text = df['text'].isna().sum()
        missing_labels = df['label'].isna().sum()
        
        print(f"   Missing text entries: {missing_text}")
        print(f"   Missing label entries: {missing_labels}")
        
        # Remove rows with missing text (essential)
        df = df.dropna(subset=['text']).reset_index(drop=True)
        
        # Handle missing labels
        if missing_labels > 0:
            # Try to infer labels or set default
            df['label'] = df['label'].fillna('unknown')
        
        # Remove empty strings
        df = df[df['text'].astype(str).str.strip() != ''].reset_index(drop=True)
        
        self.cleaning_stats['missing_removed'] = missing_text
        
        return df
    
    def _clean_text_content(self, df):
        """Clean text content"""
        print("🧽 Cleaning text content...")
        
        preprocessor = EmailPreprocessor()
        
        # Clean text using the EmailPreprocessor
        df['text'] = df['text'].apply(preprocessor.clean_text)
        
        # Remove entries that became empty after cleaning
        original_count = len(df)
        df = df[df['text'].str.strip() != ''].reset_index(drop=True)
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"   Removed {removed_count} entries that became empty after cleaning")
        
        return df
    
    def _standardize_labels(self, df):
        """Standardize label formats"""
        print("🏷️  Standardizing labels...")
        
        # Convert labels to string and normalize
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        
        # Define comprehensive label mappings
        def standardize_label(label):
            label = str(label).lower().strip()
            
            # Spam indicators
            spam_indicators = ['spam', '1', 'junk', 'unwanted', 'unsolicited', 'bad', 'malicious', 'fraud', 'phishing']
            
            # Ham indicators
            ham_indicators = ['ham', '0', 'legitimate', 'good', 'normal', 'safe', 'clean', 'ok', 'valid']
            
            # Check for spam
            if any(indicator in label for indicator in spam_indicators):
                return 'spam'
            
            # Check for ham
            if any(indicator in label for indicator in ham_indicators):
                return 'ham'
            
            # Try numeric conversion
            try:
                numeric_val = float(label)
                return 'spam' if numeric_val > 0.5 else 'ham'
            except:
                pass
            
            # If uncertain, classify based on common patterns
            if len(label) == 1 and label.isdigit():
                return 'spam' if label == '1' else 'ham'
            
            # Default to unknown for manual review
            return 'unknown'
        
        df['label'] = df['label'].apply(standardize_label)
        
        # Report label distribution
        label_dist = df['label'].value_counts()
        print(f"   Label distribution after standardization: {dict(label_dist)}")
        
        # Handle unknown labels
        unknown_count = (df['label'] == 'unknown').sum()
        if unknown_count > 0:
            print(f"   ⚠️ Found {unknown_count} unknown labels - these will be excluded from training")
            df = df[df['label'] != 'unknown'].reset_index(drop=True)
        
        self.cleaning_stats['unknown_labels_removed'] = unknown_count
        
        return df
    
    def _remove_invalid_entries(self, df):
        """Remove invalid entries"""
        print("🔍 Removing invalid entries...")
        
        original_count = len(df)
        
        # Remove texts that are too short (less than 3 characters)
        df = df[df['text'].str.len() >= 3].reset_index(drop=True)
        
        # Remove texts that contain no alphabetic characters
        df = df[df['text'].str.contains(r'[a-zA-Z]', na=False)].reset_index(drop=True)
        
        # Remove texts that are just numbers or special characters
        df = df[~df['text'].str.match(r'^[\d\s\W]+$', na=False)].reset_index(drop=True)
        
        removed_count = original_count - len(df)
        self.cleaning_stats['invalid_entries_removed'] = removed_count
        print(f"   Removed {removed_count} invalid entries")
        
        return df
    
    def _final_validation(self, df):
        """Final validation and quality checks"""
        print("✅ Performing final validation...")
        
        # Ensure we have both spam and ham samples
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            print("⚠️  Warning: Dataset contains only one label type")
            
            # If we only have one type, this might be a problem
            if len(label_counts) == 1:
                single_label = label_counts.index[0]
                print(f"   All samples are labeled as: {single_label}")
                print("   This will affect model training quality")
        
        # Check for minimum dataset size
        if len(df) < 10:
            print("⚠️  Warning: Dataset is very small (< 10 samples)")
            print("   Consider adding more data for better model performance")
        
        # Check for class imbalance
        if len(label_counts) == 2:
            ratio = label_counts.max() / label_counts.min()
            if ratio > 10:
                print(f"⚠️  Warning: High class imbalance (ratio: {ratio:.2f})")
                print("   Consider balancing the dataset for better model performance")
        
        # Final statistics
        print(f"   Final dataset size: {len(df)} samples")
        print(f"   Average text length: {df['text'].str.len().mean():.1f} characters")
        print(f"   Vocabulary size (approx): {len(set(' '.join(df['text']).split()))}")
        
        return df
    
    def _save_cleaned_dataset(self, df, output_path):
        """Save cleaned dataset to specified path"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Cleaned dataset saved to: {output_path}")
            
            # Also save as JSON for versatility
            json_path = output_path.replace('.csv', '.json')
            df.to_json(json_path, orient='records', indent=2)
            print(f"💾 Also saved as JSON: {json_path}")
            
            # Save metadata
            metadata = {
                'original_format': self.detected_format,
                'original_encoding': self.detected_encoding,
                'cleaning_stats': self.cleaning_stats,
                'final_shape': df.shape,
                'columns': list(df.columns),
                'label_distribution': df['label'].value_counts().to_dict()
            }
            
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"📋 Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error saving cleaned dataset: {str(e)}")
    
    def _print_cleaning_summary(self):
        """Print comprehensive cleaning summary"""
        print("\n" + "="*60)
        print("📊 DATASET CLEANING SUMMARY")
        print("="*60)
        print(f"Original format: {self.detected_format}")
        if self.detected_encoding:
            print(f"Detected encoding: {self.detected_encoding}")
        print(f"Original rows: {self.cleaning_stats['original_rows']:,}")
        print(f"Final rows: {self.cleaning_stats['final_rows']:,}")
        print(f"Rows removed: {self.cleaning_stats['rows_removed']:,}")
        print(f"Removal percentage: {self.cleaning_stats['removal_percentage']:.2f}%")
        print("\nDetailed removals:")
        print(f"  - Duplicates: {self.cleaning_stats.get('duplicates_removed', 0):,}")
        print(f"  - Missing values: {self.cleaning_stats.get('missing_removed', 0):,}")
        print(f"  - Unknown labels: {self.cleaning_stats.get('unknown_labels_removed', 0):,}")
        print(f"  - Invalid entries: {self.cleaning_stats.get('invalid_entries_removed', 0):,}")
        print("="*60)

def process_your_dataset(dataset_path, output_dir=None):
    """
    Main function to process your dataset
    """
    print("🚀 PROCESSING YOUR DATASET")
    print("="*50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    # Initialize cleaner
    cleaner = UniversalDatasetCleaner()
    
    # Set output path
    if output_dir is None:
        output_dir = 'data/processed'
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_dataset.csv')
    
    # Process the dataset
    cleaned_dataset = cleaner.detect_format_and_clean(dataset_path, output_path)
    
    if cleaned_dataset is not None:
        print(f"\n✅ SUCCESS! Your dataset has been processed")
        print(f"📊 Final dataset shape: {cleaned_dataset.shape}")
        print(f"📁 Cleaned dataset saved to: {output_path}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned data:")
        print(cleaned_dataset.head())
        
        return output_path
    else:
        print("❌ Dataset processing failed")
        return None

if __name__ == "__main__":
    # Example usage - replace with your dataset path
    dataset_path = "data/raw/spamhamdata.csv"  # Put your dataset here
    processed_path = process_your_dataset(dataset_path)
