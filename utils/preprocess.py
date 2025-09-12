import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Comprehensive text preprocessing function
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', ' ', text)
        
        # Remove money patterns ($100, £50, etc.)
        text = re.sub(r'[$£€¥]\d+(?:[\.,]\d+)?', ' money ', text)
        
        # Remove excessive exclamation marks
        text = re.sub(r'!{2,}', '!', text)
        
        # Remove excessive question marks
        text = re.sub(r'\?{2,}', '?', text)
        
        # Remove excessive dots
        text = re.sub(r'\.{3,}', '.', text)
        
        # Remove numbers but keep important patterns
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords, single characters, and stem
        filtered_words = []
        for word in words:
            if (word not in stop_words and 
                len(word) > 1 and 
                word.isalpha()):
                stemmed_word = ps.stem(word)
                filtered_words.append(stemmed_word)
        
        # Join words back
        processed_text = ' '.join(filtered_words)
        
        # Return empty string if no valid words remain
        return processed_text.strip() if processed_text.strip() else ''
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return ''

def get_text_stats(text):
    """
    Get basic statistics about the text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary containing text statistics
    """
    if not isinstance(text, str):
        text = str(text)
    
    words = text.split()
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }

# Test function
def test_preprocessing():
    """Test the preprocessing function with various inputs"""
    test_cases = [
        "Hello! How are you doing today???",
        "URGENT!!! Win $1000 NOW!!! Call 123-456-7890",
        "Visit https://example.com for more info",
        "Contact us at support@company.com",
        "Meeting at 2:30 PM in room 101",
        "",
        123,
        None
    ]
    
    print("Testing preprocessing function:")
    for i, test_case in enumerate(test_cases, 1):
        processed = preprocess_text(test_case)
        print(f"{i}. Original: {test_case}")
        print(f"   Processed: {processed}")
        print(f"   Stats: {get_text_stats(str(test_case) if test_case is not None else '')}")
        print("-" * 50)

if __name__ == "__main__":
    test_preprocessing()
