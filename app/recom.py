from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback
import uvicorn

# Global storage for model artifacts
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading recommendation system...")
    try:
        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / 'model.pkl'
        DATA_PATH = BASE_DIR / 'user_item_matrix.csv'

        if MODEL_PATH.exists() and DATA_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                ml_models["model"] = pickle.load(f)
            ml_models["user_item_matrix"] = pd.read_csv(DATA_PATH, index_col='user_id')
            print("Setup complete! Model and data loaded.")
        else:
            print(f"WARNING: Files not found at {BASE_DIR}")
            print("Run train_model.py first.")
            ml_models["model"] = None
            ml_models["user_item_matrix"] = None
    except Exception as e:
        print(f"Error during setup: {e}")
        ml_models["model"] = None
        ml_models["user_item_matrix"] = None
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get('/')
def home():
    return {
        'status': 'online',
        'endpoints': {
            '/recommend/<user_id>': 'Get recommendations for a specific user',
        },
        'sample_usage': 'Try /recommend/1 to get recommendations for user 1'
    }

@app.get('/recommend/{user_id}')
def recommend(user_id: int):
    model = ml_models.get("model")
    user_item_matrix = ml_models.get("user_item_matrix")

    # Check if model and data are loaded
    if model is None or user_item_matrix is None:
        return JSONResponse(content={
            'error': 'Model or data not loaded. Run train_model.py first!',
            'status': 'error'
        }, status_code=503)
        
    try:
        # Check if user exists
        if user_id not in user_item_matrix.index:
            return JSONResponse(content={
                'error': f'User ID {user_id} not found',
                'status': 'error'
            }, status_code=404)
            
        # Get user's ratings
        user_ratings = user_item_matrix.loc[user_id]
        
        # Find similar items for items the user rated highly (rating >= 4)
        liked_items = user_ratings[user_ratings >= 4].index.tolist()
        
        if not liked_items:
            # If no highly rated items, recommend most popular items
            popular_items = user_item_matrix.mean().nlargest(5).index.tolist()
            return {
                'user_id': user_id,
                'recommendations': popular_items,
                'strategy': 'popular_items',
                'status': 'success'
            }
            
        # Get recommendations using the model
        _, indices = model.kneighbors(user_item_matrix[liked_items].T.mean().values.reshape(1, -1))
        # Map indices back to item names
        recommendations = user_item_matrix.columns[indices[0]].tolist()
            
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'strategy': 'similar_items',
            'status': 'success'
        }
            
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(content={
            'error': str(e),
            'traceback': tb,
            'status': 'error'
        }, status_code=500)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    # Instructions
    print("\nStarting API server...")
    print("- Use Ctrl+C to stop the server")
    print(f"- Visit http://localhost:{port}/ for API information")
    print(f"- Visit http://localhost:{port}/docs for interactive API documentation")
    print(f"- Try http://localhost:{port}/recommend/1 for sample recommendations")
    
    # Start the FastAPI server
    uvicorn.run(app, host='0.0.0.0', port=port)