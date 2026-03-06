import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

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
                ngram_range=(2, 5),
                lowercase=True
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
    ("classifier", LinearSVC(C=1.0))
])


# Train model
model.fit(X_train, y_train)


# Evaluate
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", accuracy)


# Save trained model
with open("models/gender_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved at models/gender_model.pkl")