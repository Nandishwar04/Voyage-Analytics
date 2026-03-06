import pickle

# Load trained model
model = pickle.load(open("models/gender_model.pkl", "rb"))


def predict_gender(name, age):

    # Prepare input
    data = {
        "name": [name.lower()],
        "age": [age]
    }

    import pandas as pd
    df = pd.DataFrame(data)

    prediction = model.predict(df)

    return prediction[0]