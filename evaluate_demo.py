#!/usr/bin/env python3
"""
Demonstration of the evaluation module for clustering quality assessment.

This script shows how to use the ClusterEvaluator to assess clustering quality,
constraint satisfaction, and track progress across iterations.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Import our evaluation module
from src.evaluation import ClusterEvaluator, evaluate_engine, print_evaluation_report
from src.engine import ClusterRefinementEngine


def demo_basic_evaluation():
    """Demonstrate basic clustering evaluation with synthetic data."""
    print("🔬 CLUSTERING EVALUATION DEMO")
    print("=" * 50)

    # Generate synthetic clustering data
    np.random.seed(42)
    X, true_labels = make_blobs(n_samples=100, centers=4, cluster_std=1.5, random_state=42)

    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    predicted_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Evaluate clustering quality
    evaluator = ClusterEvaluator()

    print("\n📊 Evaluating Clustering Quality...")
    cluster_metrics = evaluator.evaluate_clustering(X, predicted_labels, centroids)

    print(f"Silhouette Score: {cluster_metrics.silhouette_score}")
    print(f"Calinski-Harabasz Score: {cluster_metrics.calinski_harabasz_score}")
    print(f"Davies-Bouldin Score: {cluster_metrics.davies_bouldin_score}")

    print("\n🏗️ Cluster Statistics:")
    for i, (size, variance) in enumerate(zip(cluster_metrics.cluster_sizes, cluster_metrics.cluster_variances)):
        print(f"  Cluster {i}: {size} items, variance: {variance:.3f}")


def demo_constraint_evaluation():
    """Demonstrate constraint satisfaction evaluation."""
    print("\n" + "=" * 50)
    print("🎯 CONSTRAINT SATISFACTION DEMO")
    print("=" * 50)

    # Create mock clustering results
    np.random.seed(42)
    n_samples = 50
    labels = np.random.randint(0, 3, n_samples)  # 3 clusters

    # Define some constraints
    must_links = [(0, 1), (5, 6), (10, 11)]  # Should be in same cluster
    cannot_links = [(0, 25), (15, 30)]  # Should be in different clusters

    # Manually set some constraints to be satisfied
    labels[0] = labels[1]  # Satisfy must_link (0,1)
    labels[0] = 0  # Put item 0 in cluster 0
    labels[25] = 1  # Put item 25 in different cluster from item 0

    evaluator = ClusterEvaluator()
    constraint_metrics = evaluator.evaluate_constraint_satisfaction(labels, must_links, cannot_links)

    print(f"Must-link satisfaction: {constraint_metrics.must_link_satisfaction:.1%}")
    print(f"Cannot-link satisfaction: {constraint_metrics.cannot_link_satisfaction:.1%}")
    print(f"Total constraints: {constraint_metrics.total_must_links + constraint_metrics.total_cannot_links}")


def demo_stability_evaluation():
    """Demonstrate stability tracking across iterations."""
    print("\n" + "=" * 50)
    print("🔄 STABILITY TRACKING DEMO")
    print("=" * 50)

    evaluator = ClusterEvaluator()

    # First iteration
    labels1 = np.array([0, 0, 1, 1, 2, 2])
    stability1 = evaluator.evaluate_stability(labels1)
    print(f"First iteration - Assignment stability: {stability1.assignment_stability:.1%}")

    # Second iteration with some changes
    labels2 = np.array([0, 1, 1, 1, 2, 2])  # Changed one assignment
    stability2 = evaluator.evaluate_stability(labels2)
    print(f"Second iteration - Assignment stability: {stability2.assignment_stability:.1%}")

    # Third iteration (back to original)
    labels3 = np.array([0, 0, 1, 1, 2, 2])
    stability3 = evaluator.evaluate_stability(labels3)
    print(f"Third iteration - Assignment stability: {stability3.assignment_stability:.1%}")


def demo_full_engine_evaluation():
    """Demonstrate evaluation with the full clustering engine."""
    print("\n" + "=" * 50)
    print("🚀 FULL ENGINE EVALUATION DEMO")
    print("=" * 50)

    # Create sample texts for clustering
    texts = [
        "Action movie with explosions and car chases",
        "Romantic comedy with light-hearted jokes",
        "Horror film with scary monsters",
        "Documentary about space exploration",
        "Western with gunfights and horses",
        "Thriller with detectives and mystery",
        "Animated family movie for kids",
        "Superhero film with special effects",
        "Drama about friendship and betrayal",
        "Sci-fi adventure with time travel"
    ] * 2  # Duplicate for more samples

    # Initialize engine
    print("Initializing clustering engine...")
    engine = ClusterRefinementEngine(
        texts=texts,
        n_clusters=3,
        device="cpu",
        use_projection_head=False,
        num_heads=1
    )

    # Evaluate initial clustering
    print("\n📈 Initial Clustering Evaluation:")
    initial_report = print_evaluation_report(engine, iteration=0)

    # Apply some feedback
    print("\n💬 Applying User Feedback...")

    # Add some must-links
    feedback1 = "Items 0 and 10 should be together (both action movies)"
    print(f"User: {feedback1}")
    reply1 = engine.chat(feedback1)
    print(f"Assistant: {reply1}")

    # Add some cannot-links
    feedback2 = "Items 0 and 5 should NOT be together (action vs thriller)"
    print(f"\nUser: {feedback2}")
    reply2 = engine.chat(feedback2)
    print(f"Assistant: {reply2}")

    # Evaluate after feedback
    print("\n📈 Post-Feedback Evaluation:")
    final_report = print_evaluation_report(engine, iteration=1)

    # Compare improvements
    print("\n🔍 COMPARING RESULTS:")
    evaluator = ClusterEvaluator()
    improvements = evaluator.compare_reports(initial_report, final_report)

    print("Key Improvements:")
    for metric, change in improvements.items():
        if isinstance(change, (int, float)):
            direction = "📈" if change > 0 else "📉"
            print(f"  {direction} {metric}: {change:+.3f}")


if __name__ == "__main__":
    demo_basic_evaluation()
    demo_constraint_evaluation()
    demo_stability_evaluation()
    demo_full_engine_evaluation()

    print("\n" + "=" * 60)
    print("✅ EVALUATION DEMO COMPLETE")
    print("=" * 60)
    print("\n📚 Key Takeaways:")
    print("• Use evaluate_engine() for quick assessment")
    print("• Track stability across iterations")
    print("• Monitor constraint satisfaction rates")
    print("• Compare reports to measure progress")
    print("• Use print_evaluation_report() for detailed output")
