import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/svd_recommendation_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    # Your model was saved as a dict with these keys:
    # 'U', 'sigma', 'Vt', 'preds_df', 'user_ratings_mean'
    preds_df = saved["preds_df"]
    print(f"[INFO] SVD model loaded. Users: {preds_df.shape[0]}, Hotels: {preds_df.shape[1]}")
    print(f"[INFO] Known user IDs: {list(preds_df.index)}")
    print(f"[INFO] Known hotel IDs: {list(preds_df.columns)}")

except FileNotFoundError:
    preds_df = None
    print("[WARN] svd_recommendation_model.pkl not found.")


def get_recommendations(user_id: int, item_ids: list, top_n: int = 5) -> list:
    """
    Generate top-N hotel recommendations for a user.

    Uses the pre-computed preds_df matrix from SVD decomposition.
    user_id must exist in the training data index.
    item_ids filters recommendations to only those hotel IDs.

    Args:
        user_id:  Target user ID (must be in preds_df index)
        item_ids: List of candidate hotel IDs to score (pass [] to get all)
        top_n:    How many top results to return

    Returns:
        List of {"item_id": int, "estimated_rating": float} sorted descending
    """
    if preds_df is None:
        raise RuntimeError("SVD model is not loaded. Ensure svd_recommendation_model.pkl exists.")

    # Check if user exists in the model
    if user_id not in preds_df.index:
        available = list(preds_df.index)
        raise ValueError(
            f"User ID {user_id} not found in model. "
            f"Available user IDs are: {available}"
        )

    # Get this user's predicted ratings row
    user_predictions = preds_df.loc[user_id]

    # If specific item_ids were requested, filter to only those
    # (ignore any item_ids not known to the model)
    if item_ids:
        known_items = [i for i in item_ids if i in preds_df.columns]
        if not known_items:
            available_hotels = list(preds_df.columns)
            raise ValueError(
                f"None of the requested item IDs {item_ids} exist in the model. "
                f"Known hotel IDs are: {available_hotels}"
            )
        user_predictions = user_predictions[known_items]

    # Sort by predicted rating descending and take top_n
    top_predictions = user_predictions.sort_values(ascending=False).head(top_n)

    results = [
        {
            "item_id": int(hotel_id),
            "estimated_rating": round(float(rating), 4)
        }
        for hotel_id, rating in top_predictions.items()
    ]

    return results