import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("🏦 Предсказание оттока клиента")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (years)", 0, 10, 5)
    with col2:
        balance = st.number_input("Balance", 0.0, value=50000.0)
        num_products = st.number_input("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
        estimated_salary = st.number_input("Estimated Salary", 0.0, value=100000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            pred = result["churn_prediction"]
            if pred == 1:
                st.error(f"⚠️ Высокий риск оттока! Вероятность: {prob:.2%}")
            else:
                st.success(f"✅ Низкий риск оттока. Вероятность: {prob:.2%}")
        else:
            st.error(f"Ошибка API: {response.text}")
    except Exception as e:
        st.error(f"Не удалось соединиться с сервером: {e}")