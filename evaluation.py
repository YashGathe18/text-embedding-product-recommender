from sklearn.metrics import pairwise_distances
import numpy as np

def evaluate_recommendations(model, test_data):
    """Evaluate the recommendation system by calculating accuracy metrics."""
    # Example evaluation: use a validation dataset and calculate average cosine similarity
    similarities = pairwise_distances(test_data['embeddings'], model.embeddings, metric='cosine')
    avg_similarity = np.mean(similarities)
    return avg_similarity
