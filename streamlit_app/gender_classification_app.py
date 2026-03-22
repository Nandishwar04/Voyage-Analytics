import streamlit as st
import requests
import pandas as pd 

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="ML Prediction Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Prediction Platform")
st.caption("Gender Classification · Flight Price Prediction · SVD Recommendations")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API Base URL", value=API_BASE)
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.status_code == 200:
                st.success(f"✅ API Online")
            else:
                st.error(f"❌ Status {r.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach API: {e}")

tab1, tab2, tab3 = st.tabs(
    ["👤 Gender Classification", "✈️ Flight Price Prediction", "🎯 SVD Recommendations"]
)

# ══════════════════════════════════════════════
# TAB 1 – Gender
# ══════════════════════════════════════════════
with tab1:
    st.subheader("👤 Gender Classification")

    # Load real names from your dataset
    @st.cache_data
    def load_users():
        df = pd.read_csv("data/raw/users.csv")
        return df

    users_df = load_users()

    col1, col2 = st.columns(2)
    with col1:
        # Searchable dropdown — user can type to filter the 1339 names
        selected_name = st.selectbox(
            "Select Name",
            options=users_df["name"].dropna().sort_values().unique(),
            help="Start typing to search through names"
        )

        # Auto-fill age from dataset when name is selected
        matched_row = users_df[users_df["name"] == selected_name].iloc[0]
        auto_age = int(matched_row["age"])

    with col2:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=auto_age,
            help="Auto-filled from dataset, you can also edit manually"
        )
        # Show extra info from the row
        st.info(f"🏢 Company: **{matched_row['company']}** &nbsp;|&nbsp; Code: **{matched_row['code']}**")

    if st.button("Predict Gender", key="btn_gender"):
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{api_url}/predict_gender",
                    json={"name": selected_name, "age": age},
                    timeout=5,
                )
                if response.status_code == 200:
                    result = response.json()
                    gender = result["predicted_gender"]
                    actual_gender = matched_row["gender"]
                    icon = "🙎‍♀️" if "female" in gender.lower() else "🙎‍♂️"

                    st.success(f"{icon} Predicted Gender: **{gender}**")

                    # Show comparison with actual gender from dataset
                    if gender.lower() == actual_gender.lower():
                        st.success(f"✅ Matches actual gender in dataset: **{actual_gender}**")
                    else:
                        st.warning(f"⚠️ Actual gender in dataset: **{actual_gender}**")

                    st.json(result)
                else:
                    st.error(f"API Error {response.status_code}: {response.json()}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is the Flask server running?")

# ══════════════════════════════════════════════
# TAB 2 – Flight Price
# ══════════════════════════════════════════════
with tab2:
    st.subheader("✈️ Flight Price Prediction")
    st.write("Predict Brazilian domestic flight prices.")

    col1, col2 = st.columns(2)

    CITIES = [
        "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
        "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
        "Salvador (BH)", "Sao Paulo (SP)"
    ]

    with col1:
        travel_code  = st.number_input("Travel Code",  min_value=1, value=1001, step=1)
        user_code    = st.number_input("User Code",    min_value=1, value=201,  step=1)
        from_city    = st.selectbox("From (Departure City)", CITIES)
        to_city      = st.selectbox("To (Arrival City)",     CITIES)
        flight_type  = st.selectbox("Flight Type", ["economic", "firstClass", "premium"])

    with col2:
        agency    = st.selectbox("Agency", ["CloudFy", "FlyingDrops", "Rainbow"])
        time      = st.number_input("Flight Time (hours)", min_value=0.5, max_value=24.0, value=2.5, step=0.5)
        distance  = st.number_input("Distance (km)",       min_value=100, max_value=10000, value=1500, step=50)
        date      = st.date_input("Travel Date")

    if st.button("Predict Price", key="btn_flight"):
        if from_city == to_city:
            st.warning("Departure and arrival city cannot be the same.")
        else:
            with st.spinner("Predicting..."):
                try:
                    payload = {
                        "travel_code": int(travel_code),
                        "user_code":   int(user_code),
                        "time":        float(time),
                        "distance":    float(distance),
                        "year":        date.year,
                        "month":       date.month,
                        "day":         date.day,
                        "from_city":   from_city,
                        "to_city":     to_city,
                        "flight_type": flight_type,
                        "agency":      agency,
                    }
                    response = requests.post(
                        f"{api_url}/predict_flight_price",
                        json=payload,
                        timeout=5,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"💰 Estimated Price: **$ {result['predicted_price_usd']:,.2f}**")
                        st.json(result)
                    else:
                        st.error(f"API Error {response.status_code}: {response.json()}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Is the Flask server running?")

# ══════════════════════════════════════════════
# TAB 3 – Recommendations
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🎯 SVD-Based Hotel Recommendations")
    st.write("Get personalised hotel recommendations based on predicted ratings.")

    col1, col2 = st.columns(2)
    with col1:
        user_id = st.selectbox(
            "User ID",
            [1, 2, 3],
            help="These are the users the model was trained on."
        )
        top_n = st.slider("Top N Recommendations", min_value=1, max_value=4, value=3)

    with col2:
        item_ids_input = st.text_area(
            "Candidate Hotel IDs (comma-separated)",
            value="101, 102, 103, 104",
            help="Only hotel IDs the model was trained on will work: 101, 102, 103, 104"
        )

    if st.button("Get Recommendations", key="btn_reco"):
        try:
            item_ids = [int(x.strip()) for x in item_ids_input.split(",") if x.strip()]
        except ValueError:
            st.error("Hotel IDs must be integers separated by commas.")
            item_ids = []

        if item_ids:
            with st.spinner("Fetching recommendations..."):
                try:
                    payload = {
                        "user_id":  user_id,
                        "item_ids": item_ids,
                        "top_n":    top_n
                    }
                    response = requests.post(
                        f"{api_url}/recommend",
                        json=payload,
                        timeout=5,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        recs = result["recommendations"]
                        st.success(f"🏨 Top {len(recs)} Hotels for User {user_id}")

                        import pandas as pd
                        df = pd.DataFrame(recs)
                        df.index += 1
                        df.columns = ["Hotel ID", "Predicted Rating"]
                        st.dataframe(df, use_container_width=True)
                        st.json(result)
                    else:
                        st.error(f"API Error {response.status_code}: {response.json()}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Is the Flask server running?")