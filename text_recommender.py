import unittest
from src.recommender import ProductRecommender

class TestRecommender(unittest.TestCase):
    def test_recommend(self):
        recommender = ProductRecommender(embeddings, product_data)
        recommendations = recommender.recommend("Sample product description", top_n=5)
        self.assertEqual(len(recommendations), 5)

if __name__ == "__main__":
    unittest.main()
