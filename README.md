# Product Recommendation System Based on Text Embeddings

This repository contains a product recommendation system that uses text embeddings to recommend similar products based on their descriptions.

## Features

- Uses pre-trained language models (Sentence Transformers) to generate text embeddings for product descriptions.
- Computes similarity between product descriptions using cosine similarity.
- Recommends the most similar products to the user.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/product-recommendation-system.git
    cd product-recommendation-system
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset (if not included):
    - Place your dataset (e.g., `products.csv`) inside the `data/` directory.

## Usage

1. **Data Preprocessing**: Run the `data_preprocessing.ipynb` notebook to load and clean the product data.
2. **Train Model**: Train the recommendation model by running `model_training.ipynb`.
3. **Generate Recommendations**: Use the functions in `recommender.py` to generate product recommendations based on input product descriptions.

## Example

```python
from src.recommender import ProductRecommender

# Initialize the recommender system
recommender = ProductRecommender()

# Recommend products based on a sample product description
product_description = "Wireless Bluetooth Headphones with Noise Cancellation"
recommended_products = recommender.recommend(product_description, top_n=5)

print(recommended_products)
