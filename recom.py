from flask import Flask, jsonify
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback

app = Flask(__name__)

print("Loading recommendation system...")

# Load the model and data
try:
    # Load the trained model
    print("Loading model from model.pkl...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the user-item matrix
    print("Loading user ratings from user_item_matrix.csv...")
    user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col='user_id')
    
    print(f"Setup complete!")
    print(f"- Loaded user-item matrix with shape: {user_item_matrix.shape}")
    print(f"- Model loaded successfully")
    
except Exception as e:
    print("\nError during setup:")
    print(f"{str(e)}")
    print("\nPlease ensure you have run train_model.py first to create:")
    print("1. user_item_matrix.csv - The sample rating data")
    print("2. model.pkl - The trained model")
    model = None
    user_item_matrix = None

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'endpoints': {
            '/recommend/<user_id>': 'Get recommendations for a specific user',
        },
        'sample_usage': 'Try /recommend/1 to get recommendations for user 1'
    })

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    # Check if model and data are loaded
    if model is None or user_item_matrix is None:
        return jsonify({
            'error': 'Model or data not loaded. Run train_model.py first!',
            'status': 'error'
        }), 503
        
    try:
        # Check if user exists
        if user_id not in user_item_matrix.index:
            return jsonify({
                'error': f'User ID {user_id} not found',
                'status': 'error'
            }), 404
            
        # Get user's ratings
        user_ratings = user_item_matrix.loc[user_id]
        
        # Find similar items for items the user rated highly (rating >= 4)
        liked_items = user_ratings[user_ratings >= 4].index.tolist()
        
        if not liked_items:
            # If no highly rated items, recommend most popular items
            popular_items = user_item_matrix.mean().nlargest(5).index.tolist()
            return jsonify({
                'user_id': user_id,
                'recommendations': popular_items,
                'strategy': 'popular_items',
                'status': 'success'
            })
            
        # Get recommendations using the model
        _, indices = model.kneighbors(user_item_matrix[liked_items].T.mean().values.reshape(1, -1))
        recommendations = indices[0].tolist()
            
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'strategy': 'similar_items',
            'status': 'success'
        })
            
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'traceback': tb,
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Instructions
    print("\nStarting API server...")
    print("- Use Ctrl+C to stop the server")
    print("- Visit http://127.0.0.1:5000/ for API information")
    print("- Try http://127.0.0.1:5000/recommend/1 for sample recommendations")
    
    # Start the Flask server
    app.run(host='127.0.0.1', port=5000, debug=True)