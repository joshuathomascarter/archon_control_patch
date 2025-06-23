# --- QISKIT & NUMPY IMPORTS ---
from qiskit import QuantumCircuit              # Core class for building quantum circuits
from qiskit_aer import Aer                     # Qiskit's high-performance simulator backend
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt                # Used for saving circuit and histogram plots
import numpy as np                             # Numerical computations (like pi)
import numpy.random as npr                     # For random noise simulation

# --- CORE FUNCTION ---
def generate_quantum_override_signal():
    """
    Simulates a noisy entangled 2-qubit system and returns a binary override signal.
    If decoherence is strong enough (high |11⟩ probability), trigger the override.
    """

    # STEP 1: Initialize a 2-qubit quantum circuit with 2 classical bits for measurement
    qc = QuantumCircuit(2, 2)

    # STEP 2: Create entanglement — Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    qc.h(0)        # Put qubit 0 into superposition
    qc.cx(0, 1)    # Entangle qubit 0 with qubit 1 using CNOT

    # STEP 3: Inject simulated noise using a randomized U3 gate on qubit 0
    theta, phi, lam = npr.uniform(0, 2 * np.pi, 3)
    qc.u(theta, phi, lam, 0)  # Apply general unitary noise to only one qubit

    # STEP 4: Measure both qubits, mapping to classical bits
    qc.measure([0, 1], [0, 1])

    # STEP 5: Simulate the circuit execution using Aer (QASM simulator = shot-based)
    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(qc, shots=1024)           # Execute 1024 times to get statistics
    result = job.result()
    counts = result.get_counts()                # e.g., {'00': 500, '11': 200, ...}

    # STEP 6: Analyze measurement counts to decide if override signal should trigger
    probability_11 = counts.get('11', 0) / 1024 # Compute P(|11⟩)
    threshold = 0.1                             # 10% threshold for override activation
    override_signal = 1 if probability_11 > threshold else 0

    return qc, counts, override_signal, probability_11

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Run the function and unpack results
    qc, counts, override_signal, prob_11 = generate_quantum_override_signal()

    # Print human-readable summary of simulation
    print(f"Simulation Counts: {counts}")
    print(f"Probability of |11⟩: {prob_11:.4f}")
    print(f"Quantum Override Signal: {override_signal}")

    # STEP 7: Save a visual of the circuit diagram to file
    try:
        qc.draw("mpl")  # Matplotlib drawer
        plt.title("Quantum Circuit Diagram")
        plt.savefig("quantum_override_sim.png")
        plt.close()
        print("Circuit diagram saved to quantum_override_sim.png")
    except Exception as e:
        print(f"Error saving circuit diagram: {e}")
        print(qc.draw("text"))  # Fallback to ASCII

    # STEP 8: Save histogram of the simulation results
    try:
        fig = plot_histogram(counts, figsize=(8, 6))
        plt.title("Measurement Results")
        plt.savefig("quantum_histogram.png")
        plt.close(fig)
        print("Histogram saved to quantum_histogram.png")
    except Exception as e:
        print(f"Error saving histogram: {e}")
