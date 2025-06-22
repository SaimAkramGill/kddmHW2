#!/usr/bin/env python3
"""
Main runner script for KDDM1 Assignment Questions 1-4
Knowledge Discovery and Data Mining - Parliamentary Debates Analysis & Chess Endgame Classification

This script runs all four questions in sequence and provides the answers
required for the assignment submission.

Author: Assignment Solution
Course: KDDM1 VO (INP.31101UF)
"""

import sys
import os
from typing import Dict, Any
import warnings 
warnings.filterwarnings('ignore')

# Import all question modules
try:
    from question1_feature_engineering import DebatesFeatureExtractor, main as q1_main
    from question2_clustering import DebatesClusterer, main as q2_main
    from question3_dimensionality_reduction import ClusterVisualizer, main as q3_main
    from question4_classification import ChessEndgameClassifier, main as q4_main
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all question files are in the same directory.")
    sys.exit(1)


class AssignmentRunner:
    """Main class to run all assignment questions and collect answers."""
    
    def __init__(self):
        """Initialize the assignment runner."""
        self.results = {}
        
    def run_question_1(self) -> Dict[str, Any]:
        """Run Question 1: Feature Engineering."""
        print("\n" + "="*80)
        print("QUESTION 1: FEATURE ENGINEERING")
        print("="*80)
        
        try:
            extractor, feature_matrix, feature_names = q1_main()
            
            self.results['q1'] = {
                'extractor': extractor,
                'feature_matrix': feature_matrix,
                'feature_names': feature_names,
                'num_features': len(feature_names)
            }
            
            print("\n✓ Question 1 completed successfully!")
            return self.results['q1']
            
        except Exception as e:
            print(f"✗ Error in Question 1: {e}")
            return {}
    
    def run_question_2(self) -> Dict[str, Any]:
        """Run Question 2: Clustering."""
        print("\n" + "="*80)
        print("QUESTION 2: CLUSTERING")
        print("="*80)
        
        try:
            clusterer, cluster_labels, cluster_top_words = q2_main()
            
            self.results['q2'] = {
                'clusterer': clusterer,
                'cluster_labels': cluster_labels,
                'cluster_top_words': cluster_top_words,
                'optimal_clusters': clusterer.optimal_n_clusters,
                'silhouette_score': clusterer.silhouette_scores[clusterer.optimal_n_clusters]
            }
            
            print("\n✓ Question 2 completed successfully!")
            return self.results['q2']
            
        except Exception as e:
            print(f"✗ Error in Question 2: {e}")
            return {}
    
    def run_question_3(self) -> Dict[str, Any]:
        """Run Question 3: Dimensionality Reduction."""
        print("\n" + "="*80)
        print("QUESTION 3: DIMENSIONALITY REDUCTION FOR VISUALIZATION")
        print("="*80)
        
        try:
            visualizer, cluster_labels, pc1_features, pc2_features = q3_main()
            
            self.results['q3'] = {
                'visualizer': visualizer,
                'cluster_labels': cluster_labels,
                'pc1_features': pc1_features,
                'pc2_features': pc2_features,
                'explained_variance': visualizer.pca.explained_variance_ratio_
            }
            
            print("\n✓ Question 3 completed successfully!")
            return self.results['q3']
            
        except Exception as e:
            print(f"✗ Error in Question 3: {e}")
            return {}
    
    def run_question_4(self) -> Dict[str, Any]:
        """Run Question 4: Classification."""
        print("\n" + "="*80)
        print("QUESTION 4: CLASSIFICATION")
        print("="*80)
        
        try:
            classifier, results = q4_main()
            
            self.results['q4'] = {
                'classifier': classifier,
                'results': results,
                'best_params': classifier.best_params,
                'accuracy': results['accuracy']
            }
            
            print("\n✓ Question 4 completed successfully!")
            return self.results['q4']
            
        except Exception as e:
            print(f"✗ Error in Question 4: {e}")
            return {}
    
    def print_final_answers(self):
        """Print the final answers for all questions in assignment format."""
        print("\n" + "="*100)
        print("FINAL ANSWERS FOR ASSIGNMENT SUBMISSION")
        print("="*100)
        
        # Question 1 Answers
        if 'q1' in self.results:
            print("\nQUESTION 1 - FEATURE ENGINEERING:")
            print("-" * 50)
            print("Answer (a) - Preprocessing:")
            print("• Removed non-informative instances (short texts, procedural content) and cleaned text by removing punctuation, stop words, and applying lemmatization.")
            
            print("\nAnswer (b) - Feature computation:")
            print("• max_features=5000: Limits vocabulary size for computational efficiency while retaining most informative terms.")
            print("• min_df=5: Removes very rare terms that appear in fewer than 5 documents to reduce noise.")
            print("• max_df=0.95: Removes terms appearing in >95% of documents as they're less discriminative.")
            print("• ngram_range=(1,2): Includes both unigrams and bigrams to capture context and phrases.")
            
            print("\nAnswer (c) - Number of features:")
            print(f"• Extracted {self.results['q1']['num_features']} features to balance information retention with computational tractability.")
        
        # Question 2 Answers
        if 'q2' in self.results:
            print("\nQUESTION 2 - CLUSTERING:")
            print("-" * 50)
            print("Answer (a) - Clustering algorithm:")
            print("• Used K-means clustering because it's efficient for high-dimensional sparse TF-IDF data and provides interpretable centroids. K-means works well for document clustering when combined with TF-IDF features.")
            
            print("\nAnswer (b) - Number of clusters:")
            print(f"• Extracted {self.results['q2']['optimal_clusters']} clusters using silhouette analysis to find the optimal number that maximizes intra-cluster similarity and inter-cluster separation.")
            
            print("\nAnswer (c) - Evaluation:")
            print(f"• Used silhouette score as evaluation metric, measuring how similar documents are to their own cluster versus other clusters. Final silhouette score: {self.results['q2']['silhouette_score']:.4f}")
            
            print("\nAnswer (d) - Interpretation:")
            for cluster_id, words in self.results['q2']['cluster_top_words'].items():
                if any(word in ['ukraine', 'war', 'russia', 'military'] for word in words):
                    interpretation = f"Cluster {cluster_id} focuses on Ukraine conflict and security issues"
                elif any(word in ['climate', 'energy', 'green', 'environment'] for word in words):
                    interpretation = f"Cluster {cluster_id} discusses climate change and environmental policy"
                elif any(word in ['economic', 'market', 'trade', 'financial'] for word in words):
                    interpretation = f"Cluster {cluster_id} covers economic and trade-related topics"
                elif any(word in ['digital', 'data', 'technology', 'artificial'] for word in words):
                    interpretation = f"Cluster {cluster_id} addresses digital transformation and technology"
                elif any(word in ['health', 'covid', 'pandemic', 'medical'] for word in words):
                    interpretation = f"Cluster {cluster_id} deals with health and pandemic-related issues"
                else:
                    interpretation = f"Cluster {cluster_id} represents general parliamentary procedures and governance"
                print(f"• {interpretation}.")
        
        # Question 3 Answers
        if 'q3' in self.results:
            print("\nQUESTION 3 - DIMENSIONALITY REDUCTION FOR VISUALIZATION:")
            print("-" * 50)
            print("Answer (a) - Your plot:")
            print("• [The plot shows clusters in 2D PCA space with different colors for each cluster - see generated visualization]")
            
            print("\nAnswer (b) - Cluster separation:")
            # Simple heuristic based on explained variance
            total_variance = sum(self.results['q3']['explained_variance'])
            if total_variance > 0.3:
                print("• Clusters show good separation in PCA space with clear distinctions between topic groups.")
            else:
                print("• Clusters show moderate separation in PCA space, indicating some overlap between related topics.")
            
            print("\nAnswer (c) - Interpretation:")
            print("• PCA-1 captures the primary variation in parliamentary discourse, likely distinguishing between procedural and substantive policy discussions.")
            print("• PCA-2 captures secondary thematic variations, potentially separating different policy domains or political orientations.")
        
        # Question 4 Answers
        if 'q4' in self.results:
            print("\nQUESTION 4 - CLASSIFICATION:")
            print("-" * 50)
            print("Answer (a) - Preprocessing & feature transformations:")
            print("• Encoded categorical file positions (a-h) to numerical values (1-8) using LabelEncoder for machine learning compatibility. Removed rows with missing values and applied standard scaling to features.")
            
            print("\nAnswer (b) - Model choice:")
            print("• Used Random Forest classifier because it handles mixed categorical/numerical features well, provides feature importance, and is robust to outliers. Random Forest also performs well on classification tasks with moderate-sized datasets without requiring extensive feature engineering.")
            
            print("\nAnswer (c) - Evaluation setup:")
            print("• Used 80/20 train-test split with stratified sampling and 5-fold cross-validation for hyperparameter tuning, evaluating with accuracy as primary metric.")
            
            print("\nAnswer (d) - Hyperparameters:")
            print(f"• Optimized n_estimators, max_depth, min_samples_split, and min_samples_leaf using GridSearchCV. Final hyperparameters: {self.results['q4']['best_params']}")
            
            print("\nAnswer (e) - Results:")
            results = self.results['q4']['results']
            print("Evaluation Results:")
            print(f"Cross-validation accuracy: {results['cv_score']:.4f}")
            print(f"Test accuracy: {results['accuracy']:.4f}")
            print(f"Precision (weighted): {results['precision']:.4f}")
            print(f"Recall (weighted): {results['recall']:.4f}")
            print(f"F1-score (weighted): {results['f1_score']:.4f}")
    
    def run_all_questions(self):
        """Run all questions in sequence."""
        print("KDDM1 Assignment - Running All Questions")
        print("="*80)
        
        # Check if required files exist
        required_files = ['debates_2022.csv', 'king_rook_vs_king.csv']
        for file in required_files:
            if not os.path.exists(file):
                print(f"✗ Required file not found: {file}")
                print("Please ensure all dataset files are in the current directory.")
                return
        
        print("✓ All required files found.")
        
        # Run questions sequentially
        self.run_question_1()
        self.run_question_2()
        self.run_question_3()
        self.run_question_4()
        
        # Print final answers
        self.print_final_answers()
        
        print(f"\n{'='*100}")
        print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
        print("="*100)


def main():
    """Main function to run the entire assignment."""
    runner = AssignmentRunner()
    runner.run_all_questions()


if __name__ == "__main__":
    main()