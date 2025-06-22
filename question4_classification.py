import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ChessEndgameClassifier:
    """Classifies chess endgame outcomes using machine learning."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features (file positions) to numerical values."""
        df_encoded = df.copy()
        
        # Encode file positions (a-h) to numbers (1-8)
        file_columns = ['white_king_file', 'white_rook_file', 'black_king_file']
        
        for col in file_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        return df_encoded
    
    def _transform_target_variable(self, target_series: pd.Series) -> pd.Series:
        """Transform target variable according to assignment requirements."""
        def map_depth_to_level(depth_str):
            if pd.isna(depth_str):
                return 0
            
            depth_str = str(depth_str).strip().lower()
            
            if depth_str == 'draw':
                return 0
            elif depth_str in ['zero', '0', 'one', '1', 'two', '2', 'three', '3', 'four', '4']:
                return 1
            elif depth_str in ['five', '5', 'six', '6', 'seven', '7', 'eight', '8']:
                return 2
            elif depth_str in ['nine', '9', 'ten', '10', 'eleven', '11', 'twelve', '12']:
                return 3
            elif depth_str in ['thirteen', '13', 'fourteen', '14', 'fifteen', '15', 'sixteen', '16']:
                return 4
            else:
                # Try to parse as number
                try:
                    num = int(depth_str)
                    if num == 0:
                        return 0
                    elif 1 <= num <= 4:
                        return 1
                    elif 5 <= num <= 8:
                        return 2
                    elif 9 <= num <= 12:
                        return 3
                    elif 13 <= num <= 16:
                        return 4
                    else:
                        return 0  # Default to draw for unexpected values
                except:
                    return 0
        
        return target_series.apply(map_depth_to_level)
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the chess endgame dataset.
        
        Args:
            df: Raw dataset
            
        Returns:
            Tuple of (features, target)
        """
        print("Preprocessing chess endgame data...")
        
        # Handle missing values
        df_clean = df.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")
        
        # Encode categorical features
        df_encoded = self._encode_categorical_features(df_clean)
        
        # Extract features (positions)
        feature_columns = ['white_king_file', 'white_king_rank', 
                          'white_rook_file', 'white_rook_rank', 
                          'black_king_file', 'black_king_rank']
        
        X = df_encoded[feature_columns].values
        self.feature_names = feature_columns
        
        # Transform target variable
        y = self._transform_target_variable(df_clean['white_depth_of_win'])
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:")
        for level, count in pd.Series(y).value_counts().sort_index().items():
            print(f"  Level {level}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y.values
    
    def train_model(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   test_size: float = 0.2,
                   cv_folds: int = 5) -> Dict:
        """
        Train Random Forest classifier with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation results
        """
        print("Splitting data and training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features (important for distance-based algorithms, optional for RF)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Performing hyperparameter optimization with cross-validation...")
        
        # Initialize base model
        rf = RandomForestClassifier(random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_score': grid_search.best_score_,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Draw', 'Quick Win', 'Medium Win', 'Long Win', 'Very Long Win'],
                   yticklabels=['Draw', 'Quick Win', 'Medium Win', 'Long Win', 'Very Long Win'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        print("Feature Importance:")
        for feature, importance in sorted_importance.items():
            print(f"  {feature}: {importance:.4f}")
        
        return sorted_importance


def main():
    """Main function to demonstrate chess endgame classification."""
    print("Chess Endgame Classification")
    print("=" * 40)
    
    # Load dataset
    print("Loading chess endgame dataset...")
    df = pd.read_csv('king_rook_vs_king.csv')
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize classifier
    classifier = ChessEndgameClassifier()
    
    # Preprocess data
    X, y = classifier.preprocess_data(df)
    
    # Train model
    print(f"\nTraining Random Forest classifier...")
    results = classifier.train_model(X, y, test_size=0.2, cv_folds=5)
    
    # Print detailed results
    print(f"\nDetailed Classification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # Show feature importance
    importance = classifier.get_feature_importance()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    features = list(importance.keys())
    values = list(importance.values())
    
    plt.barh(features, values)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Chess Endgame Classification')
    plt.tight_layout()
    plt.show()
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()
    
    # Answers for the assignment
    print("\n" + "="*60)
    print("ANSWERS FOR QUESTION 4:")
    print("="*60)
    
    print("\nAnswer (a) - Preprocessing & feature transformations:")
    print("• Encoded categorical file positions (a-h) to numerical values (1-8) using LabelEncoder for machine learning compatibility. Removed rows with missing values and applied standard scaling to features.")
    
    print("\nAnswer (b) - Model choice:")
    print("• Used Random Forest classifier because it handles mixed categorical/numerical features well, provides feature importance, and is robust to outliers. Random Forest also performs well on classification tasks with moderate-sized datasets without requiring extensive feature engineering.")
    
    print("\nAnswer (c) - Evaluation setup:")
    print("• Used 80/20 train-test split with stratified sampling and 5-fold cross-validation for hyperparameter tuning, evaluating with accuracy as primary metric.")
    
    print("\nAnswer (d) - Hyperparameters:")
    print(f"• Optimized n_estimators, max_depth, min_samples_split, and min_samples_leaf using GridSearchCV. Final hyperparameters: {classifier.best_params}")
    
    print("\nAnswer (e) - Results:")
    print("Evaluation Results:")
    print(f"Cross-validation accuracy: {results['cv_score']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Precision (weighted): {results['precision']:.4f}")
    print(f"Recall (weighted): {results['recall']:.4f}")
    print(f"F1-score (weighted): {results['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print("          Draw  Quick  Medium  Long  VLong")
    cm = results['confusion_matrix']
    for i, row in enumerate(cm):
        labels = ['Draw   ', 'Quick  ', 'Medium ', 'Long   ', 'VLong  ']
        print(f"{labels[i]} {' '.join(f'{val:5d}' for val in row)}")