import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from src.data.preprocess_users import load_users_data
from src.features.gender_features import prepare_features


# Load dataset
df = load_users_data()

# Prepare features
X, y = prepare_features(df)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature transformation
preprocessor = ColumnTransformer(
    transformers=[
        (
            "name_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),       # Reduced from (2,5) — less memorization
                lowercase=True,
                min_df=2,                 # Ignore very rare n-grams (reduces overfitting)
                max_features=5000,        # Cap features to avoid memorizing unique names
                sublinear_tf=True         # Log-scale TF to reduce dominance of frequent n-grams
            ),
            "name"
        ),
        (
            "age_scaler",
            StandardScaler(),
            ["age"]
        )
    ]
)


# Build pipeline
model = Pipeline([
    ("features", preprocessor),
    ("classifier", LinearSVC(
        C=0.1,          # Lower C = stronger regularization = less overfitting
        max_iter=2000
    ))
])


# Train model
model.fit(X_train, y_train)


# Evaluate on test set
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Cross-validation to check generalization (5-fold)
# If CV score ≈ test accuracy, model is generalizing well
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\nCross-Val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# Overwrite existing model file (no new file created)
model_path = "models/gender_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel updated at {model_path}")