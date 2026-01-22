import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path

# Create a sample user-item matrix
n_users = 100
n_items = 50
np.random.seed(42)

# Create random ratings (0-5 scale)
user_item_matrix = pd.DataFrame(
    np.random.randint(0, 6, size=(n_users, n_items)),
    columns=[f'item_{i}' for i in range(n_items)]
)
user_item_matrix.index.name = 'user_id'

# Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Save the user-item matrix
user_item_matrix.to_csv(BASE_DIR / 'user_item_matrix.csv')

# Train a simple item-based recommendation model
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(user_item_matrix.T)  # transpose for item-based similarity

# Save the model
with open(BASE_DIR / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Created and saved:")
print("- Sample user-item matrix: user_item_matrix.csv")
print("- Recommendation model: model.pkl")