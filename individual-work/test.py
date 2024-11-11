import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and StandardScaler
model_path = 'Trained_models/RandomForestClassifier.pkl'
scaler_path = 'Trained_models/scaler.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Function to process input data
def process_input(oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, transaction_type):
    # Create DataFrame from input data
    data = pd.DataFrame([[0, oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest]],
                         columns=['amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest'])

    # Create one-hot encoding for transaction types
    cash_in = 1 if transaction_type == 'cash_in' else 0
    cash_out = 1 if transaction_type == 'cash_out' else 0
    debit = 1 if transaction_type == 'debit' else 0
    payment = 1 if transaction_type == 'payment' else 0
    transfer = 1 if transaction_type == 'transfer' else 0

    # Append transaction type data to DataFrame
    transaction_data = pd.DataFrame([[cash_in, cash_out, debit, payment, transfer]],
                                     columns=['cash_in', 'cash_out', 'debit', 'payment', 'transfer'])

    # Combine data
    processed_data = pd.concat([data, transaction_data], axis=1)

    return processed_data

# Function to predict transaction type
def predict_transaction(oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, transaction_type):
    # Transform input data
    processed_data = process_input(oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, transaction_type)

    # Scale the data using the original DataFrame
    scaled_data = scaler.transform(processed_data)

    # Predict
    prediction = model.predict(scaled_data)

    # Output result
    if prediction[0] == 1:
        return 'Ця транзакція є шахрайською!'
    else:
        return 'Ця транзакція є безпечною.'

# Transaction examples
transactions = [
    (170136.00, 160296.36, 0.00, 0.00, 'payment'),         # Likely safe (from your data)
    (21249.00, 19384.72, 0.00, 0.00, 'payment'),           # Likely safe (from your data)
    (181.00, 0.00, 0.00, 0.00, 'transfer'),                # Likely fraudulent (from your data)
    (181.00, 0.00, 21182.00, 0.00, 'cash_out'),            # Likely fraudulent (from your data)
    (41554.00, 29885.86, 0.00, 0.00, 'payment'),           # Likely safe (from your data)
    (339682.13, 0.00, 0.00, 339682.13, 'cash_out'),        # Likely fraudulent (from your data)
    (6311409.28, 0.00, 0.00, 0.00, 'transfer'),            # Likely fraudulent (from your data)
]

# Iterate through transactions and make predictions
for oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, transaction_type in transactions:
    result = predict_transaction(oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, transaction_type)
    print(f"Transaction: ({oldBalanceOrig}, {newBalanceOrig}, {oldBalanceDest}, {newBalanceDest}, '{transaction_type}') -> {result}")