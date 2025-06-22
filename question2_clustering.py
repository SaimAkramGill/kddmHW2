import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import from Question 1
from question1_feature_engineering import DebatesFeatureExtractor


class DebatesClusterer:
    """Clusters parliamentary debate transcripts using K-means algorithm."""
    
    def __init__(self, 
                 n_clusters_range: Tuple[int, int] = (3, 15),
                 random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            n_clusters_range: Range of cluster numbers to evaluate
            random_state: Random state for reproducibility
        """
        self.n_clusters_range = n_clusters_range
        self.random_state = random_state
        
        self.optimal_n_clusters: Optional[int] = None
        self.clusterer: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.silhouette_scores: Dict[int, float] = {}
        self.feature_names: Optional[List[str]] = None
        
    def _evaluate_clustering(self, 
                           feature_matrix: np.ndarray, 
                           n_clusters: int) -> float:
        """Evaluate clustering quality for given number of clusters."""
        kmeans = KMeans(n_clusters=n_clusters, 
                       random_state=self.random_state, 
                       n_init=10,
                       max_iter=300)
        
        labels = kmeans.fit_predict(feature_matrix)
        
        # Use silhouette score as evaluation metric
        score = silhouette_score(feature_matrix, labels)
        return score
    
    def find_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        print("Evaluating different numbers of clusters...")
        
        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            score = self._evaluate_clustering(feature_matrix, n_clusters)
            self.silhouette_scores[n_clusters] = score
            print(f"n_clusters={n_clusters}: silhouette_score={score:.4f}")
        
        # Find optimal number of clusters
        self.optimal_n_clusters = max(self.silhouette_scores.keys(), 
                                    key=lambda k: self.silhouette_scores[k])
        
        print(f"\nOptimal number of clusters: {self.optimal_n_clusters}")
        print(f"Best silhouette score: {self.silhouette_scores[self.optimal_n_clusters]:.4f}")
        
        return self.optimal_n_clusters
    
    def fit(self, 
            feature_matrix: np.ndarray, 
            feature_names: List[str],
            n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Fit K-means clustering to the feature matrix.
        
        Args:
            feature_matrix: TF-IDF feature matrix
            feature_names: Names of features
            n_clusters: Number of clusters (if None, find optimal)
            
        Returns:
            Cluster labels
        """
        self.feature_names = feature_names
        
        # Find optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(feature_matrix)
        else:
            self.optimal_n_clusters = n_clusters
        
        # Fit final clustering model
        print(f"\nFitting K-means with {n_clusters} clusters...")
        self.clusterer = KMeans(n_clusters=n_clusters, 
                              random_state=self.random_state,
                              n_init=10,
                              max_iter=300)
        
        self.cluster_labels = self.clusterer.fit_predict(feature_matrix)
        
        # Calculate final evaluation score
        final_score = silhouette_score(feature_matrix, self.cluster_labels)
        print(f"Final silhouette score: {final_score:.4f}")
        
        return self.cluster_labels
    
    def get_cluster_top_words(self, 
                            feature_matrix: np.ndarray, 
                            n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each cluster based on TF-IDF centroids."""
        if self.clusterer is None:
            raise ValueError("Must fit the clustering model first")
        
        cluster_top_words = {}
        
        for cluster_id in range(self.optimal_n_clusters):
            # Get cluster centroid
            centroid = self.clusterer.cluster_centers_[cluster_id]
            
            # Get top feature indices
            top_indices = centroid.argsort()[-n_words:][::-1]
            
            # Get corresponding feature names
            top_words = [self.feature_names[i] for i in top_indices]
            cluster_top_words[cluster_id] = top_words
        
        return cluster_top_words
    
    def plot_silhouette_analysis(self):
        """Plot silhouette scores for different numbers of clusters."""
        if not self.silhouette_scores:
            print("No silhouette scores available. Run find_optimal_clusters first.")
            return
        
        plt.figure(figsize=(10, 6))
        n_clusters_list = list(self.silhouette_scores.keys())
        scores = list(self.silhouette_scores.values())
        
        plt.plot(n_clusters_list, scores, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=self.optimal_n_clusters, color='r', linestyle='--', 
                   label=f'Optimal k={self.optimal_n_clusters}')
        
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_cluster_summary(self) -> Dict[int, Dict]:
        """Get summary statistics for each cluster."""
        if self.cluster_labels is None:
            raise ValueError("Must fit the clustering model first")
        
        cluster_counts = Counter(self.cluster_labels)
        
        summary = {}
        for cluster_id in range(self.optimal_n_clusters):
            summary[cluster_id] = {
                'size': cluster_counts[cluster_id],
                'percentage': cluster_counts[cluster_id] / len(self.cluster_labels) * 100
            }
        
        return summary


def main():
    """Main function to demonstrate clustering."""
    # First, extract features using Question 1
    print("Step 1: Feature Extraction")
    print("-" * 30)
    
    # Load dataset
    df = pd.read_csv('debates_2022.csv')
    
    # Extract features
    extractor = DebatesFeatureExtractor(
        max_features=5000,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    feature_matrix, feature_names = extractor.fit_transform(df)
    
    print(f"\nStep 2: Clustering")
    print("-" * 30)
    
    # Initialize clusterer
    clusterer = DebatesClusterer(n_clusters_range=(3, 12))
    
    # Fit clustering
    cluster_labels = clusterer.fit(feature_matrix, feature_names)
    
    print(f"\nStep 3: Results Analysis")
    print("-" * 30)
    
    # Get cluster summaries
    cluster_summary = clusterer.get_cluster_summary()
    print("\nCluster sizes:")
    for cluster_id, info in cluster_summary.items():
        print(f"Cluster {cluster_id}: {info['size']} documents ({info['percentage']:.1f}%)")
    
    # Get top words for each cluster
    cluster_top_words = clusterer.get_cluster_top_words(feature_matrix, n_words=10)
    
    print("\nTop words per cluster:")
    for cluster_id, words in cluster_top_words.items():
        print(f"\nCluster {cluster_id}: {', '.join(words)}")
    
    # Plot silhouette analysis
    clusterer.plot_silhouette_analysis()
    
    return clusterer, cluster_labels, cluster_top_words


if __name__ == "__main__":
    clusterer, cluster_labels, cluster_top_words = main()
    
    # Answers for the assignment
    print("\n" + "="*60)
    print("ANSWERS FOR QUESTION 2:")
    print("="*60)
    
    print("\nAnswer (a) - Clustering algorithm:")
    print("• Used K-means clustering because it's efficient for high-dimensional sparse TF-IDF data and provides interpretable centroids. K-means works well for document clustering when combined with TF-IDF features.")
    
    print(f"\nAnswer (b) - Number of clusters:")
    print(f"• Extracted {clusterer.optimal_n_clusters} clusters using silhouette analysis to find the optimal number that maximizes intra-cluster similarity and inter-cluster separation.")
    
    print("\nAnswer (c) - Evaluation:")
    print(f"• Used silhouette score as evaluation metric, measuring how similar documents are to their own cluster versus other clusters. Final silhouette score: {clusterer.silhouette_scores[clusterer.optimal_n_clusters]:.4f}")
    
    print("\nAnswer (d) - Interpretation:")
    for cluster_id, words in cluster_top_words.items():
        # Create interpretation based on top words
        top_3_words = words[:3]
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