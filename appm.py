import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Insurance Prediction Analysis",
    page_icon="💛",
    layout="wide"
)

# ---------------------------------------------------
# PROFESSIONAL YELLOW THEME CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f9fafb;
}

/* Container Card */
.main-card {
    background-color: white;
    padding: 40px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}

/* Headings */
h1 {
    color: #d4a017;
    font-weight: 700;
}

h3 {
    color: #444444;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #f7c948, #fadb5f);
    color: black;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #fadb5f, #f7c948);
}

/* Result Box */
.result-box {
    background: #fff8dc;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #333;
    border-left: 6px solid #f7c948;
}

/* Metric Style */
.metric-box {
    background: #fffdf5;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #f7e6a1;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

data = load_data()

# Encoding categorical variables
data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
data.replace({'region': {'southeast': 0, 'southwest': 1,
                         'northeast': 2, 'northwest': 3}}, inplace=True)

X = data.drop(columns='charges')
Y = data['charges']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

r2 = r2_score(Y_test, model.predict(X_test))

# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("💛 Insurance Cost Prediction System")
st.markdown("### Predict Medical Insurance Charges Using Machine Learning")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", 18, 100, 30)
    sex = st.selectbox("⚧ Gender", ["male", "female"])
    bmi = st.number_input("⚖ BMI (Body Mass Index)", 10.0, 50.0, 25.0)

with col2:
    children = st.slider("👶 Number of Children", 0, 5, 0)
    smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
    region = st.selectbox("🌍 Region", 
                          ["southeast", "southwest", "northeast", "northwest"])

# Encoding user input
sex = 0 if sex == "male" else 1
smoker = 0 if smoker == "yes" else 1
region_dict = {"southeast": 0, "southwest": 1,
               "northeast": 2, "northwest": 3}
region = region_dict[region]

st.markdown("<br>", unsafe_allow_html=True)

if st.button("💰 Predict Insurance Cost"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)

    st.markdown(f"""
        <div class="result-box">
        💵 Estimated Insurance Cost: ${prediction[0]:,.2f}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"""
        <div class="metric-box">
        📊 <b>Model Accuracy (R² Score)</b><br>
        {r2:.3f}
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="metric-box">
        🤖 <b>Model Used</b><br>
        Linear Regression
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)