import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Define constants for actions (matching Verilog 2-bit encoding)
ACTION_OK = 0    # 2'b00
ACTION_STALL = 1 # 2'b01
ACTION_FLUSH = 2 # 2'b10
ACTION_OVERRIDE = 3 # 2'b11 (LOCK in FSM, but ML predicts "override" action)

ACTION_MAP = {
    ACTION_OK: '00',
    ACTION_STALL: '01',
    ACTION_FLUSH: '10',
    ACTION_OVERRIDE: '11'
}

# Parameters for synthetic data generation
NUM_SAMPLES_PER_CLASS = 200 # Number of sequences per action type
SEQUENCE_LENGTH = 10        # Number of (entropy, chaos, ipc_var) pairs in each sequence
NUM_FEATURES = 3            # Entropy, Chaos Score, IPC Variance

# Thresholds for synthetic data labeling (aligning with your provided logic)
ENTROPY_STALL_THRESHOLD = 0.7
CHAOS_STALL_THRESHOLD = 0.6
IPC_VAR_STALL_THRESHOLD = 0.5

ENTROPY_FLUSH_THRESHOLD = 0.9
CHAOS_FLUSH_THRESHOLD = 0.8
IPC_VAR_FLUSH_THRESHOLD = 0.7

# Model filename
MODEL_FILENAME = 'model.h5'

def generate_synthetic_data(num_samples_per_class, sequence_length):
    """
    Generates synthetic sequential data for training the LSTM model with 3 features:
    entropy, chaos score, and IPC variance.
    Labels are assigned based on thresholds for the average of these features over a sequence.
    """
    X = [] # Features: sequences of (entropy, chaos, ipc_var)
    y = [] # Labels: corresponding action (OK, STALL, FLUSH, OVERRIDE)

    # Helper to map scaled values (0-1) to appropriate 16-bit ranges for better
    # simulation of real sensor values, while still using the 0-1 range for thresholds.
    def scale_to_16bit(val_0_1):
        return int(val_0_1 * 65535)

    # Generate data for ACTION_OK (low values for all features)
    for _ in range(num_samples_per_class):
        sequence = []
        entropy_series = np.random.rand(sequence_length) * 0.4 # 0-0.4 scaled
        chaos_series = np.random.rand(sequence_length) * 0.3 # 0-0.3 scaled
        ipc_var_series = np.random.rand(sequence_length) * 0.2 # 0-0.2 scaled
        for t in range(sequence_length):
            sequence.append([scale_to_16bit(entropy_series[t]),
                             scale_to_16bit(chaos_series[t]),
                             scale_to_16bit(ipc_var_series[t])])
        X.append(sequence)
        y.append(ACTION_OK)

    # Generate data for ACTION_STALL (moderate values for features)
    for _ in range(num_samples_per_class):
        sequence = []
        entropy_series = np.random.rand(sequence_length) * 0.4 + 0.3 # 0.3-0.7 scaled
        chaos_series = np.random.rand(sequence_length) * 0.4 + 0.2 # 0.2-0.6 scaled
        ipc_var_series = np.random.rand(sequence_length) * 0.4 + 0.1 # 0.1-0.5 scaled
        for t in range(sequence_length):
            sequence.append([scale_to_16bit(entropy_series[t]),
                             scale_to_16bit(chaos_series[t]),
                             scale_to_16bit(ipc_var_series[t])])
        X.append(sequence)
        y.append(ACTION_STALL)

    # Generate data for ACTION_FLUSH (high values for features)
    for _ in range(num_samples_per_class):
        sequence = []
        entropy_series = np.random.rand(sequence_length) * 0.3 + 0.6 # 0.6-0.9 scaled
        chaos_series = np.random.rand(sequence_length) * 0.3 + 0.5 # 0.5-0.8 scaled
        ipc_var_series = np.random.rand(sequence_length) * 0.3 + 0.4 # 0.4-0.7 scaled
        for t in range(sequence_length):
            sequence.append([scale_to_16bit(entropy_series[t]),
                             scale_to_16bit(chaos_series[t]),
                             scale_to_16bit(ipc_var_series[t])])
        X.append(sequence)
        y.append(ACTION_FLUSH)

    # Generate data for ACTION_OVERRIDE (very high values for features)
    for _ in range(num_samples_per_class):
        sequence = []
        entropy_series = np.random.rand(sequence_length) * 0.2 + 0.8 # 0.8-1.0 scaled
        chaos_series = np.random.rand(sequence_length) * 0.2 + 0.7 # 0.7-0.9 scaled
        ipc_var_series = np.random.rand(sequence_length) * 0.2 + 0.6 # 0.6-0.8 scaled
        for t in range(sequence_length):
            sequence.append([scale_to_16bit(entropy_series[t]),
                             scale_to_16bit(chaos_series[t]),
                             scale_to_16bit(ipc_var_series[t])])
        X.append(sequence)
        y.append(ACTION_OVERRIDE)

    X = np.array(X)
    y = np.array(y)

    # Scale the features
    scaler = MinMaxScaler()
    X_reshaped = X.reshape(-1, NUM_FEATURES)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)

    # Convert labels to one-hot encoding
    y_one_hot = to_categorical(y, num_classes=len(ACTION_MAP))

    print(f"Generated {X.shape[0]} samples.")
    print(f"Label distribution: OK={np.sum(y==0)}, STALL={np.sum(y==1)}, FLUSH={np.sum(y==2)}, OVERRIDE={np.sum(y==3)}")


    return X_scaled, y_one_hot, scaler

def build_lstm_model(input_shape, num_classes):
    """
    Builds a simple LSTM model for hazard prediction.
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    """
    Generates data, trains the LSTM model, and saves it.
    """
    print("Generating synthetic data for 3 features (Entropy, Chaos, IPC Variance)...")
    X, y, scaler = generate_synthetic_data(NUM_SAMPLES_PER_CLASS, SEQUENCE_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify to maintain class balance

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

    model = build_lstm_model((SEQUENCE_LENGTH, NUM_FEATURES), len(ACTION_MAP))
    model.summary()

    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Test Accuracy: {accuracy*100:.2f}%")

    # Save the model and the scaler (scaler is crucial for consistent prediction)
    model.save(MODEL_FILENAME)
    print(f"Model saved as '{MODEL_FILENAME}'")

    # Save scaler parameters (min and max values used for scaling)
    scaler_params = {
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'feature_range': scaler.feature_range
    }
    with open('scaler_params.npy', 'wb') as f:
        np.save(f, scaler_params)
    print("Scaler parameters saved as 'scaler_params.npy'")

def load_predictor_model():
    """
    Loads the trained LSTM model and scaler parameters.
    Returns a prediction function.
    """
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found. Please run train_and_save_model() first.")
        return None

    model = load_model(MODEL_FILENAME)
    print(f"Model '{MODEL_FILENAME}' loaded successfully.")

    # Load scaler parameters
    with open('scaler_params.npy', 'rb') as f:
        scaler_params = np.load(f, allow_pickle=True).item()

    # Reconstruct MinMaxScaler
    scaler = MinMaxScaler(feature_range=scaler_params['feature_range'])
    scaler.data_min_ = np.array(scaler_params['data_min_'])
    scaler.data_max_ = np.array(scaler_params['data_max_'])
    # Ensure data_range_ and scale_ are set, as they are used internally
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    # Handle potential division by zero if a feature has zero range
    scaler.scale_ = np.where(scaler.data_range_ != 0, scaler.feature_range[1] / scaler.data_range_, 0)


    # Create a history buffer for the LSTM input
    prediction_history = []

    def predict_action(entropy_value, chaos_score, ipc_variance):
        nonlocal prediction_history
        # Add current values to history
        prediction_history.append([entropy_value, chaos_score, ipc_variance])
        # Keep only the latest SEQUENCE_LENGTH values
        if len(prediction_history) > SEQUENCE_LENGTH:
            prediction_history = prediction_history[-SEQUENCE_LENGTH:]

        # If history is not yet full, pad with the first available value or zeros
        if len(prediction_history) < SEQUENCE_LENGTH:
            # Pad with the first available entry or zeros if history is empty
            if prediction_history:
                padding_value = prediction_history[0]
            else:
                padding_value = [0.0, 0.0, 0.0] # Default zeros for empty history
            padded_history = [padding_value] * (SEQUENCE_LENGTH - len(prediction_history)) + prediction_history
        else:
            padded_history = prediction_history

        # Scale the current sequence
        current_sequence_scaled = scaler.transform(np.array(padded_history))

        # Reshape for model prediction: (1, sequence_length, num_features)
        input_for_prediction = current_sequence_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

        # Make prediction
        probabilities = model.predict(input_for_prediction, verbose=0)[0]
        predicted_class = np.argmax(probabilities)

        return ACTION_MAP.get(predicted_class, '00') # Default to '00' (OK) if somehow invalid

    return predict_action

if __name__ == "__main__":
    # --- Step 1: Train and save the model ---
    # Run this once to generate your model.h5 and scaler_params.npy
    train_and_save_model()

    # --- Step 2: Demonstrate prediction ---
    print("\nDemonstrating prediction with loaded model:")
    predictor = load_predictor_model()

    if predictor:
        # Simulate some example (entropy, chaos, ipc_var) triplets
        test_cases = [
            (5000, 3000, 1000),    # Expected: OK
            (25000, 10000, 8000),  # Expected: STALL
            (12000, 28000, 15000), # Expected: STALL
            (45000, 35000, 25000), # Expected: FLUSH
            (30000, 50000, 40000), # Expected: FLUSH
            (60000, 62000, 55000), # Expected: OVERRIDE
            (7000, 7000, 2000),    # Expected: OK (after some history)
            (55000, 58000, 50000)  # Expected: OVERRIDE (after some history)
        ]

        print("\nRunning test cases (feeding one by one to build history):")
        for i, (e, c, ipc) in enumerate(test_cases):
            predicted_action_code = predictor(e, c, ipc)
            print(f"Input (E:{e}, C:{c}, IPC_Var:{ipc}) -> Predicted Action: {predicted_action_code}")
