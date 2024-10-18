import torch
from sklearn.metrics.pairwise import cosine_similarity

class ProductRecommender:
    def __init__(self, embeddings, product_data):
        self.embeddings = embeddings
        self.product_data = product_data

    def recommend(self, product_description, top_n=5):
        """Recommend products based on the input product description."""
        embedder = TextEmbedder()
        query_embedding = embedder.generate_embeddings([product_description])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)

        # Get top_n most similar products
        similar_indices = similarities.argsort()[0][-top_n:]
        return self.product_data.iloc[similar_indices]
