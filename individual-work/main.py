import pandas as pd
import pickle
import streamlit as st


# Load the model and StandardScaler
model_path = 'Trained_models/RandomForestClassifier.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

scaler_path = 'Trained_models/scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


# Function to process input data
def process_input(old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, transaction_type):
    data = pd.DataFrame(
        [[0, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest]],
        columns=['amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']
    )

    # Create one-hot encoding for transaction types
    transaction_types = ['cash_in', 'cash_out', 'debit', 'payment', 'transfer']
    transaction_data = {t: 1 if transaction_type == t else 0 for t in transaction_types}

    # Append transaction type data to DataFrame
    transaction_df = pd.DataFrame([transaction_data])

    # Combine data
    processed_data = pd.concat([data, transaction_df], axis=1)

    return processed_data

# Function to predict transaction type
def predict_transaction(old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, transaction_type):
    # Transform input data
    processed_data = process_input(old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, transaction_type)

    # Scale the data using the original DataFrame
    scaled_data = scaler.transform(processed_data)

    # Predict
    prediction = model.predict(scaled_data)

    # Output result
    return 'Ця транзакція є шахрайською!' if prediction[0] == 1 else 'Ця транзакція є безпечною.'

# Streamlit app
st.title("Прогнозування шахрайських транзакцій")

# Add spacing
st.write("\n\n")
st.write("\n\n")

# Create two columns for sender and recipient input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Дані відправника")
    old_balance_orig = st.number_input("Сума до транзакції (відправник):", min_value=0.0, step=0.01, key="old_balance_orig")
    new_balance_orig = st.number_input("Сума після транзакції (відправник):", min_value=0.0, step=0.01, key="new_balance_orig")

with col2:
    st.subheader("Дані отримувача")
    old_balance_dest = st.number_input("Сума до транзакції (отримувач):", min_value=0.0, step=0.01, key="old_balance_dest")
    new_balance_dest = st.number_input("Сума після транзакції (отримувач):", min_value=0.0, step=0.01, key="new_balance_dest")

# Input for transaction type
transaction_type = st.selectbox("Тип транзакції:", ['cash_in', 'cash_out', 'debit', 'payment', 'transfer'])

# Prediction button
if st.button("Прогнозувати"):
    result = predict_transaction(old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, transaction_type)
    st.success(result)