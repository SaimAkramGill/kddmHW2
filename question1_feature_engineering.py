import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from typing import Optional, Tuple, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Handles text preprocessing for parliamentary debates."""
    
    def __init__(self):
        """Initialize preprocessor with necessary NLTK components."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass  # Handle case where NLTK downloads fail
        
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            # Fallback if NLTK is not available
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean individual text by removing noise and normalizing."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words and very short words
        words = text.split()
        words = [word for word in words if len(word) > 2 and word not in self.stop_words]
        
        # Apply lemmatization if available
        if self.lemmatizer:
            try:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            except:
                pass  # Continue without lemmatization if it fails
        
        return ' '.join(words)
    
    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """Preprocess entire corpus of texts."""
        return [self.clean_text(text) for text in texts]


class DebatesFeatureExtractor:
    """Extracts TF-IDF features from parliamentary debate transcripts."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 min_df: int = 5,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize feature extractor with TF-IDF parameters.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (as fraction)
            ngram_range: Range of n-grams to consider
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.preprocessor = TextPreprocessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
    
    def _remove_non_informative_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove very short or empty talks that provide little information."""
        # Remove rows with missing or very short text
        df_clean = df.dropna(subset=['talk_text'])
        df_clean = df_clean[df_clean['talk_text'].str.len() > 50]  # At least 50 characters
        
        # Remove procedural/administrative texts
        procedural_keywords = ['session', 'resumption', 'interlude', 'musical', 'applause']
        mask = ~df_clean['talk_text'].str.lower().str.contains('|'.join(procedural_keywords), na=False)
        df_clean = df_clean[mask]
        
        return df_clean
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract TF-IDF features from debate texts.
        
        Args:
            df: DataFrame with 'talk_text' column
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Preprocessing: Remove non-informative instances
        df_clean = self._remove_non_informative_instances(df)
        print(f"After preprocessing: {len(df_clean)} instances (removed {len(df) - len(df_clean)})")
        
        # Clean texts
        cleaned_texts = self.preprocessor.preprocess_corpus(df_clean['talk_text'].tolist())
        
        # Initialize TF-IDF vectorizer with chosen parameters
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,  # Limit vocabulary size for computational efficiency
            min_df=self.min_df,             # Remove very rare terms (noise)
            max_df=self.max_df,             # Remove very common terms (less informative)
            ngram_range=self.ngram_range,   # Include unigrams and bigrams for context
            lowercase=True,                 # Already handled in preprocessing
            stop_words=None,                # Already handled in preprocessing
            token_pattern=r'\b\w+\b'       # Simple word tokenization
        )
        
        # Fit and transform the data
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        print(f"Extracted {self.feature_matrix.shape[1]} features from {self.feature_matrix.shape[0]} documents")
        
        return self.feature_matrix, self.feature_names
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N features by average TF-IDF score."""
        if self.feature_matrix is None:
            raise ValueError("Must fit the model first")
        
        avg_scores = np.mean(self.feature_matrix, axis=0)
        top_indices = np.argsort(avg_scores)[-n:][::-1]
        
        return [self.feature_names[i] for i in top_indices]


def main():
    """Main function to demonstrate feature extraction."""
    # Load the dataset
    print("Loading debates dataset...")
    df = pd.read_csv('debates_2022.csv')
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize feature extractor
    extractor = DebatesFeatureExtractor(
        max_features=5000,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    # Extract features
    print("\nExtracting TF-IDF features...")
    feature_matrix, feature_names = extractor.fit_transform(df)
    
    # Display results
    print(f"\nFeature extraction complete!")
    print(f"Shape of feature matrix: {feature_matrix.shape}")
    print(f"Number of features extracted: {len(feature_names)}")
    
    # Show top features
    print("\nTop 20 features by average TF-IDF score:")
    top_features = extractor.get_top_features(20)
    for i, feature in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}")
    
    return extractor, feature_matrix, feature_names


if __name__ == "__main__":
    extractor, feature_matrix, feature_names = main()
    
    # Answers for the assignment
    print("\n" + "="*60)
    print("ANSWERS FOR QUESTION 1:")
    print("="*60)
    
    print("\nAnswer (a) - Preprocessing:")
    print("• Removed non-informative instances (short texts, procedural content) and cleaned text by removing punctuation, stop words, and applying lemmatization.")
    
    print("\nAnswer (b) - Feature computation:")
    print("• max_features=5000: Limits vocabulary size for computational efficiency while retaining most informative terms.")
    print("• min_df=5: Removes very rare terms that appear in fewer than 5 documents to reduce noise.")
    print("• max_df=0.95: Removes terms appearing in >95% of documents as they're less discriminative.")
    print("• ngram_range=(1,2): Includes both unigrams and bigrams to capture context and phrases.")
    
    print("\nAnswer (c) - Number of features:")
    print(f"• Extracted {len(feature_names)} features to balance information retention with computational tractability.")