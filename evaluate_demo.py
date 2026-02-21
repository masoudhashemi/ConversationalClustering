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

    # 30 texts across 5 clear genres (3x each, no exact duplicates)
    texts = [
        # Action (0-2)
        "A fast-paced action movie with explosions and car chases, starring Arnold Schwarzenegger.",
        "An intense action blockbuster featuring martial arts and gunfights in Hong Kong.",
        "A high-octane action thriller with a bank heist and dramatic helicopter chase scenes.",
        # Comedy (3-5)
        "A romantic comedy with light-hearted jokes and love stories, starring Julia Roberts.",
        "A slapstick comedy about two bumbling thieves who accidentally save the day.",
        "An animated family comedy about toys coming to life, featuring Tim Allen and Tom Hanks.",
        # Horror (6-8)
        "A horror movie with jump scares and ghosts, set in an old house.",
        "A psychological horror film about a family trapped in a haunted hotel during winter.",
        "A slasher horror movie set in a summer camp with terrifying masked villains.",
        # Drama (9-11)
        "A dark drama about friendship and betrayal in a small town, directed by Martin Scorsese.",
        "An emotional drama about a pianist struggling with loss and finding redemption through music.",
        "A courtroom drama about a wrongfully convicted man fighting for justice.",
        # Sci-fi / Thriller (12-14)
        "A sci-fi thriller about time travel and paradoxes, directed by Christopher Nolan.",
        "A dystopian sci-fi film about robots gaining sentience and challenging their creators.",
        "A space exploration sci-fi adventure with astronauts stranded on a distant planet.",
        # Documentary (15-17)
        "A documentary about space exploration, narrated by David Attenborough.",
        "A nature documentary following the migration of wildebeest across the Serengeti.",
        "A historical documentary about the rise and fall of ancient civilizations.",
        # Western (18-19)
        "A western adventure with gunfights and horses, starring Clint Eastwood.",
        "A revisionist western about outlaws seeking redemption in the untamed frontier.",
        # Superhero (20-22)
        "A superhero action film with lots of fighting and special effects, starring Chris Hemsworth.",
        "A dark superhero movie exploring the moral cost of vigilante justice in Gotham City.",
        "A team-up superhero blockbuster featuring heroes from across the multiverse.",
        # Thriller / Mystery (23-24)
        "A slow-burning thriller about detectives solving a mystery, with long dialogues.",
        "A psychological thriller about an unreliable narrator hiding a dark secret.",
    ]

    print("Initializing clustering engine...")
    engine = ClusterRefinementEngine(
        texts=texts,
        n_clusters=4,
        device="cpu",
        use_projection_head=False,
        num_heads=1
    )

    print("\n📈 Initial Clustering Evaluation:")
    initial_report = print_evaluation_report(engine, iteration=0)

    print("\n💬 Applying User Feedback...")
    feedback_rounds = [
        # Structural constraints first
        "Items 0 and 2 should be together since they are both high-energy action movies.",
        "Items 6 and 5 should NOT be together - horror and comedy are totally different tones.",
        "Items 12 and 3 should NOT be together - sci-fi thrillers and romantic comedies are completely different.",
        "Items 9, 10, and 11 should be together - they are all serious dramas.",
        "Item 9 seems to be in the wrong cluster. It's a serious drama, not action.",
        # Semantic guidance
        "Emphasize genre and tone when clustering - these matter more than actor names.",
        # More structural constraints to reinforce separation
        "Items 20 and 0 should be together - superhero and action films belong together.",
        "Items 6 and 9 should NOT be together - horror and drama have very different tones.",
    ]

    for i, fb in enumerate(feedback_rounds, 1):
        print(f"\n[Round {i}] User: {fb}")
        reply = engine.chat(fb)
        print(f"Assistant: {reply}")

    # Convergence: extra training so all accumulated constraints settle
    print("\n⏳ Running convergence training (30 extra epochs)...")
    engine.train_step(epochs=30)

    print("\n📈 Post-Feedback Evaluation:")
    final_report = print_evaluation_report(engine, iteration=len(feedback_rounds))

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
