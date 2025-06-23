import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import circuit_drawer, plot_histogram
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os

def generate_quantum_override_signal():
    """
    Generates a quantum override signal based on a simulated 2-qubit entangled
    state subjected to a random noise gate (U3) and its subsequent measurement.

    The override signal is asserted (returns 1) if the probability of measuring
    both qubits in the |11⟩ state exceeds a defined threshold, simulating a
    critical deviation or "collapse" due to noise. This represents a
    quantum-enhanced signal injector.
    """
    # Create a Quantum Circuit with 2 qubits and 2 classical bits for measurement
    qc = QuantumCircuit(2, 2)

    # 1. Prepare a Bell state (|Φ+⟩ = (|00⟩ + |11⟩)/√2)
    # This establishes entanglement, which is sensitive to decoherence.
    qc.h(0)  # Apply Hadamard to qubit 0
    qc.cx(0, 1) # Apply CNOT with qubit 0 as control and qubit 1 as target

    # 2. Apply a Randomized Quantum Gate (U3) to simulate analog noise variation
    # U3(theta, phi, lambda) is a general single-qubit unitary gate.
    # We apply it to one of the entangled qubits to simulate a perturbation
    # that could lead to "decoherence" or a state collapse if severe enough.
    # The angles are randomized to represent unpredictable noise.
    theta = npr.uniform(0, 2 * np.pi)
    phi = npr.uniform(0, 2 * np.pi)
    lambda_ = npr.uniform(0, 2 * np.pi)
    qc.u(theta, phi, lambda_, 0) # Apply randomized U3 gate to qubit 0

    # 3. Measure both qubits
    # The measurement simulates observing the state after perturbation.
    qc.measure([0, 1], [0, 1])

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    shots = 1024 # Number of times to run the circuit
    job = execute(qc, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # Define the override condition: if |11⟩ state probability is high
    # This simulates a "collapse" to an error-indicating state.
    # Adjust this threshold based on desired sensitivity to quantum "noise."
    prob_11 = counts.get('11', 0) / shots # Probability of measuring '11'
    override_threshold = 0.6 # If P(|11>) > 60%, trigger override

    quantum_override = 1 if prob_11 > override_threshold else 0

    return qc, counts, quantum_override, prob_11

if __name__ == "__main__":
    # Generate the circuit and results
    circuit, counts, override_signal, prob_11 = generate_quantum_override_signal()

    print(f"Simulation Counts: {counts}")
    print(f"Probability of |11⟩: {prob_11:.4f}")
    print(f"Quantum Override Signal: {override_signal}")

    # --- Visual Outputs ---

    # 1. Circuit Diagram
    try:
        circuit_diagram_path = "quantum_override_sim.png"
        circuit.draw("mpl", filename=circuit_diagram_path, output='mpl', idle_wires=False)
        print(f"Circuit diagram saved to {circuit_diagram_path}")
    except Exception as e:
        print(f"Error saving circuit diagram: {e}")
        # Fallback for environments without matplotlib (e.g., some cloud environments)
        print(circuit.draw("text"))

    # 2. Simulation Result (Histogram)
    try:
        histogram_path = "quantum_histogram.png"
        fig = plot_histogram(counts, figsize=(8, 6))
        plt.title("Qiskit Simulation Results (Counts)")
        plt.savefig(histogram_path)
        plt.close(fig) # Close the figure to free memory
        print(f"Histogram saved to {histogram_path}")
    except Exception as e:
        print(f"Error saving histogram: {e}")
