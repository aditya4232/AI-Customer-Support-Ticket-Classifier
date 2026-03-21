"""
Text preprocessing module for the AI Customer Support Ticket Classifier.
Provides text cleaning and normalization functions.
"""

import re
import string
from typing import List


class TextCleaner:
    """Handles text cleaning and normalization operations."""
    
    def __init__(self):
        self.stop_words = self._get_stop_words()
    
    def _get_stop_words(self) -> set:
        """Return common English stop words."""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'about', 'against', 'between', 'into', 'through', 'during', 
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
            'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 
            'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 
            'wasn', 'weren', 'won', 'wouldn'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove common stop words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str, remove_stops: bool = True) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            remove_stops: Whether to remove stop words
            
        Returns:
            Fully preprocessed text
        """
        text = self.clean_text(text)
        if remove_stops:
            text = self.remove_stop_words(text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        words = self.tokenize(text)
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams


def preprocess_text(text: str) -> str:
    """
    Convenience function for quick text preprocessing.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    cleaner = TextCleaner()
    return cleaner.preprocess(text)


if __name__ == "__main__":
    # Test the text cleaner
    cleaner = TextCleaner()
    
    test_texts = [
        "I would like to upgrade my current plan to the premium package.",
        "I'm getting an error when trying to login. It says 'invalid credentials'",
        "I was charged twice for my subscription this month!!!",
    ]
    
    print("Text Preprocessing Test:")
    print("-" * 50)
    for text in test_texts:
        cleaned = cleaner.preprocess(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print()
