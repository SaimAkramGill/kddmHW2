import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import from previous questions
from question1_feature_engineering import DebatesFeatureExtractor
from question2_clustering import DebatesClusterer


class ClusterVisualizer:
    """Visualizes clustering results using PCA dimensionality reduction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize visualizer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.pca_features: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        
    def fit_transform_pca(self, 
                         feature_matrix: np.ndarray, 
                         n_components: int = 2) -> np.ndarray:
        """
        Apply PCA dimensionality reduction to feature matrix.
        
        Args:
            feature_matrix: Input feature matrix
            n_components: Number of PCA components to keep
            
        Returns:
            Transformed feature matrix
        """
        print(f"Applying PCA with {n_components} components...")
        
        # Standardize features before PCA (important for TF-IDF)
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_features = self.pca.fit_transform(scaled_features)
        
        # Print explained variance
        explained_variance = self.pca.explained_variance_ratio_
        print(f"Explained variance by components: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
        
        return self.pca_features
    
    def plot_clusters_2d(self, 
                        cluster_labels: np.ndarray, 
                        title: str = "Clusters in 2D PCA Space",
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot clusters in 2D PCA space.
        
        Args:
            cluster_labels: Cluster assignments for each data point
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.pca_features is None:
            raise ValueError("Must apply PCA first using fit_transform_pca()")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique clusters and create color palette
        unique_clusters = np.unique(cluster_labels)
        colors = sns.color_palette("husl", len(unique_clusters))
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            ax.scatter(self.pca_features[mask, 0], 
                      self.pca_features[mask, 1],
                      c=[colors[i]], 
                      label=f'Cluster {cluster_id}',
                      alpha=0.6, 
                      s=50)
        
        ax.set_xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.3f} variance)')
        ax.set_ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.3f} variance)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_pca_components(self, 
                              feature_names: List[str], 
                              n_features: int = 10) -> Tuple[List[str], List[str]]:
        """
        Analyze which features contribute most to each PCA component.
        
        Args:
            feature_names: Names of original features
            n_features: Number of top features to return per component
            
        Returns:
            Tuple of (top_features_pc1, top_features_pc2)
        """
        if self.pca is None:
            raise ValueError("Must apply PCA first")
        
        # Get feature loadings for each component
        components = self.pca.components_
        
        # Analyze first component
        pc1_weights = components[0]
        pc1_top_indices = np.argsort(np.abs(pc1_weights))[-n_features:][::-1]
        pc1_top_features = [feature_names[i] for i in pc1_top_indices]
        
        # Analyze second component
        pc2_weights = components[1]
        pc2_top_indices = np.argsort(np.abs(pc2_weights))[-n_features:][::-1]
        pc2_top_features = [feature_names[i] for i in pc2_top_indices]
        
        print("Top features for PCA Component 1:")
        for i, (feature, weight) in enumerate(zip(pc1_top_features, pc1_weights[pc1_top_indices])):
            print(f"  {i+1}. {feature}: {weight:.4f}")
        
        print("\nTop features for PCA Component 2:")
        for i, (feature, weight) in enumerate(zip(pc2_top_features, pc2_weights[pc2_top_indices])):
            print(f"  {i+1}. {feature}: {weight:.4f}")
        
        return pc1_top_features, pc2_top_features
    
    def evaluate_cluster_separation(self, 
                                  cluster_labels: np.ndarray) -> float:
        """
        Evaluate how well separated clusters are in PCA space.
        
        Args:
            cluster_labels: Cluster assignments
            
        Returns:
            Separation score (higher is better)
        """
        if self.pca_features is None:
            raise ValueError("Must apply PCA first")
        
        # Calculate inter-cluster vs intra-cluster distances
        unique_clusters = np.unique(cluster_labels)
        
        # Calculate cluster centers in PCA space
        cluster_centers = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            center = np.mean(self.pca_features[mask], axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        # Calculate average inter-cluster distance
        inter_cluster_dists = []
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                inter_cluster_dists.append(dist)
        
        avg_inter_cluster_dist = np.mean(inter_cluster_dists)
        
        # Calculate average intra-cluster distance
        intra_cluster_dists = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_points = self.pca_features[mask]
            center = cluster_centers[cluster_id]
            
            for point in cluster_points:
                dist = np.linalg.norm(point - center)
                intra_cluster_dists.append(dist)
        
        avg_intra_cluster_dist = np.mean(intra_cluster_dists)
        
        # Separation score: higher inter-cluster distance and lower intra-cluster distance is better
        separation_score = avg_inter_cluster_dist / (avg_intra_cluster_dist + 1e-10)
        
        return separation_score


def main():
    """Main function to demonstrate dimensionality reduction and visualization."""
    print("Step 1: Feature Extraction and Clustering")
    print("-" * 50)
    
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
    
    # Perform clustering
    clusterer = DebatesClusterer(n_clusters_range=(3, 10))
    cluster_labels = clusterer.fit(feature_matrix, feature_names)
    
    print(f"\nStep 2: PCA Dimensionality Reduction")
    print("-" * 50)
    
    # Initialize visualizer
    visualizer = ClusterVisualizer()
    
    # Apply PCA
    pca_features = visualizer.fit_transform_pca(feature_matrix, n_components=2)
    
    print(f"\nStep 3: Visualization and Analysis")
    print("-" * 50)
    
    # Plot clusters in 2D PCA space
    fig = visualizer.plot_clusters_2d(cluster_labels, 
                                     title="Parliamentary Debates Clusters in 2D PCA Space")
    
    # Analyze PCA components
    print("\nAnalyzing PCA components...")
    pc1_features, pc2_features = visualizer.analyze_pca_components(feature_names, n_features=10)
    
    # Evaluate cluster separation
    separation_score = visualizer.evaluate_cluster_separation(cluster_labels)
    print(f"\nCluster separation score: {separation_score:.4f}")
    
    # Additional analysis: plot explained variance
    plt.figure(figsize=(10, 6))
    explained_var_ratio = visualizer.pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
    plt.xlabel('PCA Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by PCA Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return visualizer, cluster_labels, pc1_features, pc2_features


if __name__ == "__main__":
    visualizer, cluster_labels, pc1_features, pc2_features = main()
    
    # Answers for the assignment
    print("\n" + "="*60)
    print("ANSWERS FOR QUESTION 3:")
    print("="*60)
    
    print("\nAnswer (a) - Your plot:")
    print("• [The plot shows clusters in 2D PCA space with different colors for each cluster]")
    
    # Evaluate cluster separation
    separation_score = visualizer.evaluate_cluster_separation(cluster_labels)
    
    print(f"\nAnswer (b) - Cluster separation:")
    if separation_score > 2.0:
        print("• Clusters are well separated in the PCA space with clear boundaries between different topic groups.")
    elif separation_score > 1.5:
        print("• Clusters show moderate separation in PCA space with some overlap between related topics.")
    else:
        print("• Clusters show limited separation in PCA space, indicating overlapping topics or need for more dimensions.")
    
    print("\nAnswer (c) - Interpretation:")
    
    # Interpret PC1 based on top features
    pc1_interpretation = "PCA-1 captures "
    if any(word in ['parliament', 'european', 'union', 'member'] for word in pc1_features[:5]):
        pc1_interpretation += "institutional and procedural aspects of European parliamentary discourse"
    elif any(word in ['ukraine', 'russia', 'war', 'security'] for word in pc1_features[:5]):
        pc1_interpretation += "geopolitical conflicts and security concerns"
    elif any(word in ['climate', 'energy', 'green', 'environment'] for word in pc1_features[:5]):
        pc1_interpretation += "environmental and climate policy discussions"
    else:
        pc1_interpretation += "general political discourse and policy debates"
    
    # Interpret PC2 based on top features
    pc2_interpretation = "PCA-2 captures "
    if any(word in ['economic', 'market', 'trade', 'financial'] for word in pc2_features[:5]):
        pc2_interpretation += "economic and trade-related policy dimensions"
    elif any(word in ['health', 'covid', 'pandemic', 'medical'] for word in pc2_features[:5]):
        pc2_interpretation += "health policy and pandemic response discussions"
    elif any(word in ['digital', 'technology', 'data', 'artificial'] for word in pc2_features[:5]):
        pc2_interpretation += "digital transformation and technology policy aspects"
    else:
        pc2_interpretation += "secondary thematic variations in parliamentary discussions"
    
    print(f"• {pc1_interpretation}.")
    print(f"• {pc2_interpretation}.")