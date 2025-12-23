import streamlit as st
import pandas as pd
import pickle as pk

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="centered"
)


# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #F4F6F9;
}

h1 {
    color: #0B5394;
}

.stButton > button {
    background-color: #0B5394;
    color: white;
    height: 3em;
    font-size: 18px;
    border-radius: 10px;
}

.stButton > button:hover {
    background-color: #073763;
    color: white;
}

.css-1r6slb0, .css-12w0qpk {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(11,83,148,0.15);
}
</style>
""", unsafe_allow_html=True)


# ------------------ LOAD MODEL ------------------
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# ------------------ HEADER ------------------
st.markdown(
    "<h1 style='text-align:center;'>🏦 Loan Approval Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px;'>Smart loan approval using Machine Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------ INPUT SECTION ------------------
st.markdown("### 📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dep = st.number_input("Number of Dependents", min_value=0, max_value=5, step=1)
    grad = st.selectbox("Education", ["Graduated", "Not Graduated"])
    self_emp = st.selectbox("Self Employed?", ["No", "Yes"])
    Annual_Income = st.number_input("Annual Income (₹)", min_value=0, step=50000)

with col2:
    Loan_Amount = st.number_input("Loan Amount (₹)", min_value=0, step=50000)
    Loan_Dur = st.number_input("Loan Duration (Years)", min_value=1, max_value=30, step=1)
    Cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    Assets = st.number_input("Assets Value (₹)", min_value=0, step=50000)

# ------------------ ENCODING ------------------
grad_s = 0 if grad == "Graduated" else 1
emp_s = 1 if self_emp == "Yes" else 0

# ------------------ PREDICTION ------------------
st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            no_of_dep,
            grad_s,
            emp_s,
            Annual_Income,
            Loan_Amount,
            Loan_Dur,
            Cibil,
            Assets
        ]],
        columns=[
            'no_of_dependents',
            'education',
            'self_employed',
            'income_annum',
            'loan_amount',
            'loan_term',
            'cibil_score',
            'Assets'
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Rejected**")
