import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration Parameters (Matching your Verilog module) ---
HISTORY_DEPTH = 3 # From your Verilog pattern_detector module
NUM_FLAGS = 4     # Zero, Negative, Carry, Overflow

# --- 2. Data Generation Function - MODIFIED for more scatter/randomness ---
def generate_flag_data(num_samples_normal, num_samples_anomaly, history_depth, num_flags):
    """
    Generates synthetic CPU ALU flag data, including normal and anomalous patterns.
    Flags are 0 (False) or 1 (True).

    Args:
        num_samples_normal (int): Number of normal flag sequences to generate.
        num_samples_anomaly (int): Number of anomalous flag sequences to generate.
        history_depth (int): The window size for flag history (e.g., 3 cycles).
        num_flags (int): The number of distinct ALU flags (4: Zero, Negative, Carry, Overflow).

    Returns:
        tuple: (DataFrame of features, Series of labels)
               Labels: 0 for normal, 1 for anomaly.
    """
    all_flag_sequences = []
    labels = []

    # --- Generate Normal Data with more variability ---
    print(f"Generating {num_samples_normal} normal flag sequences (more scattered)...")
    for _ in range(num_samples_normal):
        sequence = np.zeros((history_depth, num_flags))
        for h in range(history_depth):
            # Introduce slight probability variation per cycle
            p_zero = np.random.uniform(0.75, 0.95) # Zero flag more likely 0, but varies
            p_others_zero = np.random.uniform(0.6, 0.85) # Other flags might vary more

            # Zero flag (often 0, but can be 1)
            sequence[h, 0] = np.random.choice([0, 1], p=[p_zero, 1 - p_zero])
            # Negative, Carry, Overflow flags (more often 0)
            sequence[h, 1:] = np.random.choice([0, 1], size=num_flags-1, p=[p_others_zero, 1 - p_others_zero])

        # Occasionally inject "rare normal" patterns
        if np.random.rand() < 0.05: # 5% of normal samples are "rare normal"
            # Example: A sequence with more 1s than average, but not a defined anomaly
            sequence = np.random.choice([0, 1], size=(history_depth, num_flags), p=[0.5, 0.5])


        all_flag_sequences.append(sequence.flatten())
        labels.append(0)

    # --- Generate Anomalous Data with added noise/subtlety ---
    print(f"Generating {num_samples_anomaly} anomalous flag sequences (more random)...")
    for i in range(num_samples_anomaly):
        anomaly_sequence = np.zeros((history_depth, num_flags))

        if i % 4 == 0:
            # Pattern 1 (noisy): Mostly all flags high, but with a few random flips
            anomaly_sequence = np.random.choice([0, 1], size=(history_depth, num_flags), p=[0.1, 0.9])
        elif i % 4 == 1:
            # Pattern 2 (noisy): Rapid alternation, but with some noise
            for h in range(history_depth):
                for f in range(num_flags):
                    anomaly_sequence[h, f] = (h % 2) ^ np.random.choice([0, 1], p=[0.9, 0.1]) # 10% chance of flip
        elif i % 4 == 2:
            # Pattern 3 (noisy): Specific combination with some surrounding randomness
            # Assume order: Zero, Negative, Carry, Overflow
            anomaly_sequence[-1, 1] = 1 # Negative flag high in current cycle
            anomaly_sequence[-1, 2] = 1 # Carry flag high in current cycle
            anomaly_sequence[-1, 3] = 1 # Overflow flag high in current cycle
            # Add random noise to other parts of the sequence
            anomaly_sequence[0, :] = np.random.choice([0, 1], size=num_flags, p=[0.3, 0.7])
            anomaly_sequence[1, :] = np.random.choice([0, 1], size=num_flags, p=[0.7, 0.3])
            # Add small random noise to the specific anomaly
            if np.random.rand() < 0.2: anomaly_sequence[-1, 1] = 0 # 20% chance to remove negative
            if np.random.rand() < 0.2: anomaly_sequence[-1, 2] = 0 # 20% chance to remove carry
        else:
            # Pattern 4: Subtle deviation - a flag that is usually 0, suddenly becomes 1 for 2 consecutive cycles
            # e.g., Overflow flag (index 3)
            anomaly_sequence = np.random.choice([0, 1], size=(history_depth, num_flags), p=[0.8, 0.2]) # Start with normal-ish
            if history_depth >= 2:
                anomaly_sequence[-1, 3] = 1 # Current overflow high
                anomaly_sequence[-2, 3] = 1 # Previous overflow high
                # Ensure it's not normally high
                if np.random.rand() < 0.5: # 50% chance to make it a subtle one
                    anomaly_sequence[-3, 3] = 0 # Ensure prev2 overflow is low
            # Add some overall noise
            anomaly_sequence = anomaly_sequence + np.random.choice([0, 1], size=(history_depth, num_flags), p=[0.95, 0.05])
            anomaly_sequence = np.clip(anomaly_sequence, 0, 1) # Ensure binary

        all_flag_sequences.append(anomaly_sequence.flatten())
        labels.append(1)

    # Create DataFrame for features
    columns = []
    for h in range(history_depth):
        for f in range(num_flags):
            flag_name = ""
            if f == 0: flag_name = "zero"
            elif f == 1: flag_name = "negative"
            elif f == 2: flag_name = "carry"
            elif f == 3: flag_name = "overflow"
            columns.append(f"cycle_{h}_{flag_name}")

    X = pd.DataFrame(all_flag_sequences, columns=columns)
    y = pd.Series(labels)

    return X, y

# --- Generate the data ---
# Let's create a dataset with mostly normal samples and a small percentage of anomalies.
NUM_NORMAL_SAMPLES = 10000
NUM_ANOMALY_SAMPLES = 400 # Increased anomalies to make detection a bit harder and test robustness
X, y = generate_flag_data(NUM_NORMAL_SAMPLES, NUM_ANOMALY_SAMPLES, HISTORY_DEPTH, NUM_FLAGS)

print(f"\nGenerated dataset shape: {X.shape}")
print(f"Normal samples: {y[y==0].count()}")
print(f"Anomaly samples: {y[y==1].count()}")
print(f"Features (first 5 rows):\n{X.head()}")

# --- 3. Prepare Data for Isolation Forest ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Normal samples in train: {y_train[y_train==0].count()}")
print(f"Anomaly samples in train: {y_train[y_train==1].count()}")
print(f"Normal samples in test: {y_test[y_test==0].count()}")
print(f"Anomaly samples in test: {y_test[y_test==1].count()}")


# --- 4. Model Training (Isolation Forest) ---
model = IsolationForest(
    n_estimators=100,
    max_features=NUM_FLAGS * HISTORY_DEPTH, # Use all features
    contamination=NUM_ANOMALY_SAMPLES / (NUM_NORMAL_SAMPLES + NUM_ANOMALY_SAMPLES),
    random_state=42,
    verbose=0
)

print("\nTraining Isolation Forest model...")
model.fit(X_train)
print("Model training complete.")

# --- 5. Prediction and Evaluation ---

train_scores = model.decision_function(X_train)
test_scores = model.decision_function(X_test)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_pred_labels = np.where(train_predictions == -1, 1, 0)
test_pred_labels = np.where(test_predictions == -1, 1, 0)

print("\n--- Evaluation on Training Set ---")
print(classification_report(y_train, train_pred_labels))
print("Confusion Matrix (Training Set):\n", confusion_matrix(y_train, train_pred_labels))

print("\n--- Evaluation on Test Set ---")
print(classification_report(y_test, test_pred_labels))
print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, test_pred_labels))

# --- Visualize Anomaly Scores ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(train_scores[y_train == 0], color='blue', label='Normal (Train)', kde=True, stat='density', alpha=0.5)
sns.histplot(train_scores[y_train == 1], color='red', label='Anomaly (Train)', kde=True, stat='density', alpha=0.5)
plt.title('Anomaly Scores Distribution (Training Set)')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(test_scores[y_test == 0], color='blue', label='Normal (Test)', kde=True, stat='density', alpha=0.5)
sns.histplot(test_scores[y_test == 1], color='red', label='Anomaly (Test)', kde=True, stat='density', alpha=0.5)
plt.title('Anomaly Scores Distribution (Test Set)')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# --- ROC Curve and AUC ---
try:
    auc_train = roc_auc_score(y_train, -train_scores)
    auc_test = roc_auc_score(y_test, -test_scores)
    print(f"\nROC AUC Score (Training Set): {auc_train:.4f}")
    print(f"ROC AUC Score (Test Set): {auc_test:.4f}")

    fpr_train, tpr_train, _ = roc_curve(y_train, -train_scores)
    fpr_test, tpr_test, _ = roc_curve(y_test, -test_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'ROC curve (Train AUC = {auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'ROC curve (Test AUC = {auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

except ValueError:
    print("\nCannot compute ROC AUC: Only one class present in y_true, y_score. Ensure both normal and anomaly samples are present.")


print("\n--- Next Steps Discussion for Hardware Integration ---")
print("The Isolation Forest model generates an anomaly score. For hardware, you would:")
print("1. Extract the decision logic from the trained Isolation Forest (i.e., the structure of its decision trees).")
print("2. Simplify this logic into a series of comparisons and binary decisions that can be implemented using gates (AND, OR, NOT, multiplexers).")
print("3. Quantize any floating-point thresholds or feature values to fixed-point numbers that fit into your Verilog data types (e.g., 4-bit, 8-bit, 16-bit).")
print("4. The output `anomaly_detected_out` from this hardware module would then feed into your `archon_hazard_override_unit` as currently designed.")
print("This can be a significant simplification process, often involving custom code generation from the Python model or using specialized HLS (High-Level Synthesis) tools.")