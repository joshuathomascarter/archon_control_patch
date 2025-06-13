import argparse
import numpy as np
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit_aer import AerSimulator
from scipy.linalg import logm, expm

def von_neumann_entropy(density_matrix):
    """
    Calculates the von Neumann entropy of a density matrix.
    S = -Tr(rho * log2(rho))
    """
    # Ensure density_matrix is a numpy array for scipy.linalg
    density_matrix_np = np.array(density_matrix)

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(density_matrix_np)

    # Filter out eigenvalues close to zero to avoid log(0) issues
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    # Calculate von Neumann entropy
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def generate_entropy_bus(output_filename: str, num_samples: int, noisy: bool):
    """
    Generates a bus of 16-bit entropy values and saves them to a binary file.
    Entropy can be generated from quantum circuit simulation with noise or pseudo-random.

    Args:
        output_filename (str): The name of the file to save the entropy bus to.
        num_samples (int): The number of entropy samples to generate.
        noisy (bool): If True, generates entropy from a quantum circuit with noise.
                      If False, generates noise-free pseudo-random entropy.
    """
    print(f"Generating {num_samples} 16-bit entropy samples to '{output_filename}'...")

    # Max value for 16-bit unsigned integer is 2^16 - 1 = 65535
    max_16bit_val = 2**16 - 1

    entropy_values = []

    if noisy:
        print("Using quantum circuit simulation with noise for entropy generation.")
        # 1. Define a simple Qiskit circuit (1 qubit, 1 classical bit)
        qc = QuantumCircuit(1, 1)
        qc.h(0)        # Apply Hadamard gate to put qubit in superposition
        qc.measure(0, 0) # Measure the qubit

        # 2. Define a noise model (e.g., amplitude damping error)
        # Amplitude damping simulates energy loss to the environment.
        # This parameter 'gamma' controls the strength of the damping.
        amplitude_damping_param = 0.05 # Example: 5% probability of damping
        error = amplitude_damping_error(amplitude_damping_param)
        noise_model = NoiseModel()
        # Add error to all single-qubit gates (like Hadamard)
        noise_model.add_all_qubit_quantum_error(error, ['h'])

        # 3. Create an AerSimulator with the noise model
        # Using 'density_matrix' method for the simulator as it allows
        # direct access to the density matrix for entropy calculation.
        simulator = AerSimulator(method='density_matrix', noise_model=noise_model)

        for i in range(num_samples):
            # Transpile the circuit for the simulator (optimization for execution)
            transpiled_qc = transpile(qc, simulator)

            # Run the simulation without shots to get the final quantum state (density matrix)
            # The 'density_matrix' method directly gives the final density matrix.
            job = simulator.run(transpiled_qc)
            result = job.result()
            # The density matrix is returned as a list of density matrices, so take the first one.
            density_matrix = result.data(transpiled_qc)['density_matrix']

            # 4. Calculate von Neumann entropy of the density matrix
            # Max possible entropy for 1 qubit is log2(2) = 1.
            entropy = von_neumann_entropy(density_matrix)

            # 5. Scale the entropy to a 16-bit integer (0-65535)
            # Entropy values typically range from 0 (pure state) to log2(dimension) (maximally mixed state).
            # For a single qubit, max entropy is 1.0. We scale it linearly.
            # `min(max(entropy, 0), 1)` ensures the entropy is clipped between 0 and 1 before scaling.
            scaled_entropy = int(min(max(entropy, 0), 1) * max_16bit_val)

            # Ensure it fits in 16 bits (0 to 65535)
            scaled_entropy = max(0, min(scaled_entropy, 65535))
            entropy_values.append(scaled_entropy)

    else:
        print("Using classical pseudo-random generation for entropy.")
        # Seed for reproducibility, especially for noise-free generation, for consistent simulation runs.
        np.random.seed(42) # Keep for deterministic pseudo-random

        for _ in range(num_samples):
            # Generate noise-free, deterministic pseudo-random 16-bit values.
            # This is suitable for repeatable Verilog simulations and debugging,
            # as the input sequence will always be the same.
            entropy_values.append(np.random.randint(0, max_16bit_val + 1))

    # Save the entropy values to a binary file.
    try:
        with open(output_filename, 'wb') as f:
            for val in entropy_values:
                # Write as 2 bytes, big-endian, matching Verilog's expectation
                f.write(val.to_bytes(2, byteorder='big'))
        print(f"Successfully saved {num_samples} 16-bit entropy samples to '{output_filename}'.")
    except IOError as e:
        print(f"Error saving file '{output_filename}': {e}")
        exit(1)

def main():
    """
    Main function to parse command-line arguments and run the entropy generator.
    """
    parser = argparse.ArgumentParser(
        description="Generate a binary entropy bus file (entropy_bus.txt) for Verilog simulation."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_bus.txt",
        help="Output filename for the entropy bus (default: entropy_bus.txt)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of 16-bit entropy samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--quantum",
        action="store_true",
        dest="noisy", # Map --quantum flag to 'noisy' for consistency
        help="If set, generates entropy from a quantum circuit with noise (requires Qiskit). By default, generates classical pseudo-random entropy."
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    generate_entropy_bus(args.output, args.samples, args.noisy)

if __name__ == "__main__":
    main()