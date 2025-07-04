# Verilog Entropy-Aware Pipeline Control

This repository implements a Verilog-based 5-stage CPU pipeline (`IF`, `ID`, `EX`, `MEM`, `WB`) with adaptive hazard mitigation. It leverages machine learning (ML), quantum entropy detection, chaos monitoring, and analog/quantum overrides for enhanced resilience and security.

---

## 📦 Project Overview

The **ARCHON** architecture integrates:

- **Entropy-Aware FSM**  
  Dynamically adjusts pipeline states: `Normal`, `Stall`, `Flush`, and `Lock`.

- **ML Predictions**  
  Uses LSTM models to predict chaos-induced hazards.

- **Quantum Overrides**  
  High-priority lock signals triggered by entangled state measurements.

- **Testbench Suite**  
  Comprehensive testing under diverse entropy, anomaly, and override conditions.

---

## ✨ Features

- Adaptive hazard mitigation using real-time entropy and ML inputs  
- Modular Verilog design with Python-based co-simulation  
- Support for external entropy injection and quantum override triggers  
- Verbose logging for state transitions and override paths  

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Verilog Simulator (e.g., **VCS**, **ModelSim**, or **EDA Playground**)
- Python 3.x
- Python Libraries:
  - `tensorflow`
  - `numpy`
  - `qiskit` (optional, for quantum simulation)

---

### 📥 Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd <repo-name>
   
2. Install Python dependencies
- pip install tensorflow numpy qiskit
- Compile Verilog files

3. vcs -f filelist.f
▶️ Running the Project
- Generate Entropy Data
- python generate_entropy_bus.py
  
4. Run ML–Verilog Co-simulation
- python ml_verilog_cosim.py
  
5. Execute Verilog Testbench
./simv

# 📂 Files
Verilog Modules
- archon_top.v — Top-level CPU pipeline integration

- archon_top_testbench.v — Testbench with entropy and override simulation

- control_unit.v — Entropy-aware FSM and pipeline control logic

- pipeline_cpu.v — 5-stage RISC-style pipeline implementation

# Other submodules:

- instruction_ram.v

- alu_unit.v

- fsm_entropy_overlay.v

# Python Scripts
- chaos_lstm_predictor.py — Predicts hazard scores from entropy features

 - generate_entropy_bus.py — Injects runtime entropy sequences

- ml_verilog_cosim.py — Interfaces Python ML with Verilog simulations

- pattern_detector.py — Flags anomalous execution patterns

- quantum_override_circuit.py — Simulates quantum trigger inputs

# Documentation
- Paper 4/... — ModelSim results, architecture diagrams, and analysis

- quantum_override_writeup.docx — Design report on quantum logic integration

# 🧪 Usage Scenarios
- Scenario	Description
- Normal Operation	Validates baseline CPU flow and FSM transitions
- ML-Predicted Stall/Flush/Lock	Triggers adaptive hazard mitigation
- High Entropy	Simulates chaotic inputs to test system robustness
- Quantum Override	Forces immediate lock/recovery transitions via entangled state

# 🐞 Debugging Tips
- Monitor the following signals in the testbench logs:

- debug_pc_out

- debug_fsm_entropy_log_out

- override_trigger_log

- stall_flush_lock_state_trace

# 📜 License
MIT License (or specify your preferred license)

# 👥 Contributors
Joshua Carter

# 🔮 Future Work
- Expand supported instruction set for complex workloads

- Integrate real-world quantum data + live entropy sampling

- Deploy on FPGA with physical analog/quantum override circuits

- Explore formal verification for FSM robustness

📬 Want to collaborate?  
I'm looking for lab/startup partners for Fall 2025.  
Reach out via [LinkedIn](https://www.linkedin.com/in/joshua-carter-898868356/) or [joshuathomascarter@gmail.com](mailto:joshtcarter0710@gmail.com)

