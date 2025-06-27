// ===============================================================================
// ARCHON CORE BLOCK - Integrated CPU Implementation
// ===============================================================================
// Features:
// - 5-stage pipeline architecture with hazard detection and forwarding
// - Instruction memory with 16 instructions (expandable)
// - Branch Target Buffer for branch prediction
// - 8-register file with dual read ports
// - 4-bit ALU with flag outputs (Zero, Negative, Carry, Overflow)
// - Conditional branch execution based on ALU flags
// - Data forwarding to minimize pipeline pipeline_stalls
// - Flag register for branch condition evaluation
// - Chaos-Weighted Pipeline Override System for adaptive hazard mitigation (ENHANCED)
// - Pattern Detector for Higher-Order Anomaly Detection (ENHANCED)
// - INTEGRATED: External entropy input from 'entropy_bus.txt' for dynamic system adaptation
// - INTEGRATED: ML-predicted actions from 'ml_predictions.txt' to modulate FSM
// - INTEGRATED: ARCHON HAZARD OVERRIDE UNIT with fluctuating impact and cache miss awareness.
// - NEW: Entropy-Aware FSM Extension for log-ready control and visual inspection.
// ===============================================================================

// =====================================================================
// Enhanced Instruction Memory Module
// Features:
// - Stores 16 instructions (expandable to more if needed)
// - Uses a 4-bit program counter for addressing
// - Outputs the full instr_opcode for CPU execution
// - Optional reset capability with NOP instruction at PC=0
// =====================================================================

module instruction_ram(
    input wire clk,             // Clock signal (for synchronous read if needed)
    input wire reset,           // Reset signal
    input wire [3:0] pc_in,     // 4-bit Program Counter input
    output wire [15:0] instr_opcode // 16-bit instruction output
);

    // Instruction Memory (16 instructions of 16 bits each)
    reg [15:0] imem [0:15];

    initial begin
        // Initialize instruction memory with a sample program
        // This program is for demonstration. Replace with actual program.
        // Assume opcode format: [opcode (4)|rd (3)|rs1 (3)|rs2 (3)|imm (3)] for R-type/I-type
        // Or [opcode (4)|branch_target (12)] for J-type
        // Or [opcode (4)|rs1 (3)|imm (9)] for Load/Store etc.

        imem[0] = 16'h1234; // ADD R1, R2, R3 (opcode 1, rd=1, rs1=2, rs2=3) - Placeholder
        imem[1] = 16'h2452; // ADDI R4, R5, #2 (opcode 2, rd=4, rs1=5, imm=2) - Placeholder
        imem[2] = 16'h3678; // SUB R6, R7, R8 - Placeholder
        imem[3] = 16'h4891; // LD R8, (R9 + #1) - Placeholder
        imem[4] = 16'h5ABA; // ST R10, (R11 + #10) - Placeholder
        imem[5] = 16'h6CDE; // XOR R12, R13, R14 - Placeholder
        imem[6] = 16'h7F01; // BEQ R15, R0, +1 (branch if R15 == R0, to PC+1) - Placeholder
        imem[7] = 16'h8002; // JUMP PC+2 (unconditional jump) - Placeholder
        imem[8] = 16'h9123; // NOP - Placeholder
        imem[9] = 16'h0000; // NOP - Placeholder
        imem[10] = 16'h0000; // NOP - Placeholder
        imem[11] = 16'h0000; // NOP - Placeholder
        imem[12] = 16'h0000; // NOP - Placeholder
        imem[13] = 16'h0000; // NOP - Placeholder
        imem[14] = 16'h0000; // NOP - Placeholder
        imem[15] = 16'h0000; // NOP - Placeholder
    end

    // Instruction fetch logic
    assign instr_opcode = imem[pc_in];

endmodule


// ===============================================================================
// Branch Target Buffer (BTB) Module
// Features:
// - Stores predicted next PC for branches.
// - Improves pipeline performance by reducing branch prediction penalty.
// - Updates on misprediction.
// ===============================================================================
module branch_target_buffer(
    input wire clk,
    input wire reset,
    input wire [3:0] pc_in,             // Current PC to check for prediction
    input wire [3:0] branch_resolved_pc, // PC of branch instruction whose outcome is resolved
    input wire branch_resolved_pc_valid, // Indicates if branch_resolved_pc is valid
    input wire [3:0] branch_resolved_target_pc, // Actual target PC of the resolved branch
    input wire branch_resolved_taken, // Actual outcome of the resolved branch (taken/not taken)

    output wire [3:0] predicted_next_pc, // Predicted next PC
    output wire predicted_taken         // Predicted branch outcome (taken/not taken)
);

    // Simple BTB: Stores target PC for each instruction address
    // Each entry: {predicted_taken_bit, predicted_target_pc[3:0]}
    reg [4:0] btb_table [0:15]; // 16 entries, 5 bits each (1 for taken, 4 for PC)

    initial begin
        // Initialize BTB (e.g., all not taken, target PC is 0)
        for (integer i = 0; i < 16; i = i + 1) begin
            btb_table[i] = 5'b0_0000;
        end
    end

    // Prediction logic (combinational read)
    assign predicted_next_pc = btb_table[pc_in][3:0];
    assign predicted_taken = btb_table[pc_in][4];

    // Update logic (synchronous write)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (integer i = 0; i < 16; i = i + 1) begin
                btb_table[i] = 5'b0_0000;
            end
        end else begin
            if (branch_resolved_pc_valid) begin
                // Update BTB entry for the resolved branch
                btb_table[branch_resolved_pc] <= {branch_resolved_taken, branch_resolved_target_pc};
            end
        end
    end

endmodule


// =====================================================================
// Register File Module
// Features:
// - 8 4-bit registers (R0-R7)
// - R0 is hardwired to 0
// - Dual read ports for simultaneous operand fetching
// - Single write port for result write-back
// =====================================================================
module register_file(
    input wire clk,             // Clock signal for synchronous write
    input wire reset,           // Reset signal
    input wire regfile_write_enable, // Enable signal for write operation
    input wire [2:0] write_addr, // 3-bit address for write operation
    input wire [3:0] write_data, // 4-bit data to write

    input wire [2:0] read_addr1, // 3-bit address for read port 1
    input wire [2:0] read_addr2, // 3-bit address for read port 2
    output wire [3:0] read_data1, // 4-bit data from read port 1
    output wire [3:0] read_data2  // 4-bit data from read port 2
);

    // 8 registers, each 4 bits wide
    reg [3:0] registers [0:7];

    initial begin
        // Initialize all registers to 0 on startup
        for (integer i = 0; i < 8; i = i + 1) begin
            registers[i] = 4'h0;
        end
    end

    // Write operation (synchronous)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (integer i = 0; i < 8; i = i + 1) begin
                registers[i] = 4'h0;
            end
        end else if (regfile_write_enable) begin
            // R0 is hardwired to 0, so never write to it
            if (write_addr != 3'b000) begin
                registers[write_addr] <= write_data;
            end
        end
    end

    // Read operations (combinational)
    assign read_data1 = (read_addr1 == 3'b000) ? 4'h0 : registers[read_addr1]; // R0 always reads 0
    assign read_data2 = (read_addr2 == 3'b000) ? 4'h0 : registers[read_addr2]; // R0 always reads 0

endmodule


// =====================================================================
// ALU Module (Arithmetic Logic Unit)
// Features:
// - Performs basic arithmetic and logical operations.
// - Outputs 4-bit result and 4 flags (Zero, Negative, Carry, Overflow).
// =====================================================================
module alu_unit(
    input wire [3:0] alu_operand1, // First 4-bit operand
    input wire [3:0] alu_operand2, // Second 4-bit operand
    input wire [2:0] alu_op,       // 3-bit ALU operation code
                                   // 3'b000: ADD
                                   // 3'b001: SUB
                                   // 3'b010: AND
                                   // 3'b011: OR
                                   // 3'b100: XOR
                                   // 3'b101: SLT (Set Less Than)
                                   // Other codes can be defined for shifts, etc.
    output reg [3:0] alu_result,   // 4-bit result
    output reg zero_flag,          // Result is zero
    output reg negative_flag,      // Result is negative (MSB is 1)
    output reg carry_flag,         // Carry out from addition or borrow from subtraction
    output reg overflow_flag       // Signed overflow
);

    always @(*) begin
        alu_result = 4'h0;
        zero_flag = 1'b0;
        negative_flag = 1'b0;
        carry_flag = 1'b0;
        overflow_flag = 1'b0;

        case (alu_op)
            3'b000: begin // ADD
                alu_result = alu_operand1 + alu_operand2;
                carry_flag = (alu_operand1 + alu_operand2) > 4'b1111; // Check for unsigned carry out
                overflow_flag = ((!alu_operand1[3] && !alu_operand2[3] && alu_result[3]) || (alu_operand1[3] && alu_operand2[3] && !alu_result[3])); // Signed overflow
            end
            3'b001: begin // SUB (using 2's complement addition)
                alu_result = alu_operand1 - alu_operand2;
                carry_flag = (alu_operand1 >= alu_operand2); // For subtraction, carry_flag usually means no borrow
                overflow_flag = ((alu_operand1[3] && !alu_operand2[3] && !alu_result[3]) || (!alu_operand1[3] && alu_operand2[3] && alu_result[3])); // Signed overflow
            end
            3'b010: begin // AND
                alu_result = alu_operand1 & alu_operand2;
            end
            3'b011: begin // OR
                alu_result = alu_operand1 | alu_operand2;
            end
            3'b100: begin // XOR
                alu_result = alu_operand1 ^ alu_operand2;
            end
            3'b101: begin // SLT (Set Less Than)
                alu_result = ($signed(alu_operand1) < $signed(alu_operand2)) ? 4'h1 : 4'h0;
            end
            default: begin
                alu_result = 4'h0; // NOP or undefined
            end
        endcase

        // Common flag calculations
        if (alu_result == 4'h0)
            zero_flag = 1'b1;
        if (alu_result[3] == 1'b1) // Check MSB for signed negative
            negative_flag = 1'b1;
    end

endmodule


// =====================================================================
// Data Memory Module
// Features:
// - Simple synchronous read, asynchronous write data memory
// - Can be expanded to different sizes or types
// =====================================================================
module data_mem(
    input wire clk,             // Clock signal for synchronous operation
    input wire mem_write_enable, // Write enable signal
    input wire mem_read_enable,  // Read enable signal (for synchronous read)
    input wire [3:0] addr,       // 4-bit address input
    input wire [3:0] write_data, // 4-bit data to write
    output reg [3:0] read_data   // 4-bit data read
);

    reg [3:0] dmem [0:15]; // 16 entries, 4 bits each

    initial begin
        // Initialize data memory
        for (integer i = 0; i < 16; i = i + 1) begin
            dmem[i] = 4'h0;
        end
    end

    // Write operation (synchronous)
    always @(posedge clk) begin
        if (mem_write_enable) begin
            dmem[addr] <= write_data;
        }
    end

    // Read operation (synchronous, value is stable on next clock cycle)
    always @(posedge clk) begin
        if (mem_read_enable) begin
            read_data <= dmem[addr];
        }
    end

endmodule


// ======================================================================
// Quantum Entropy Detector Module (Simplified Placeholder)
// Features:
// - Simulates a very basic "quantum entropy" or "chaos" level.
// - This is a conceptual module; a real one would involve complex quantum state measurements.
// - Output `entropy_value` represents disorder or uncertainty.
// ======================================================================
module quantum_entropy_detector(
    input wire clk,
    input wire reset,
    input wire [3:0] instr_opcode, // Example: Opcode can influence entropy (from IF/ID)
    input wire [3:0] alu_result,   // Example: ALU result can influence entropy (from EX/MEM)
    input wire zero_flag,          // Example: ALU flags can influence entropy (from EX/MEM)
    // ... other internal CPU signals that could affect quantum state ...
    output reg [7:0] entropy_score_out // CHANGED to 8-bit to match fsm_entropy_overlay
);

    // Placeholder: Entropy value increases with complex/branching instructions
    // and decreases with NOPs or simple operations.
    // In a real Archon-like system, this would be derived from actual quantum
    // measurements or a complex internal quantum state model.
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            entropy_score_out <= 8'h00;
        end else begin
            // Simple heuristic: increase entropy on non-NOP, non-trivial ALU ops
            // and based on how 'unexpected' an ALU result might be.
            // Using 4 MSBs of 16-bit instr_opcode as actual opcode
            if (instr_opcode != 4'h9) begin // If not a NOP (assuming 4'h9 is NOP opcode)
                if (alu_result == 4'h0 && !zero_flag) begin // An "unexpected" zero result (not explicitly set)
                    entropy_score_out <= entropy_score_out + 8'h10; // Larger jump for anomaly
                end else if (entropy_score_out < 8'hFF) begin // Prevent overflow
                    entropy_score_out <= entropy_score_out + 8'h01;
                end
            end else begin
                // Reduce entropy during NOPs or idle cycles
                if (entropy_score_out > 8'h00)
                    entropy_score_out <= entropy_score_out - 8'h01;
            end
        end
    end
endmodule


// ======================================================================
// Chaos Detector Module (Simplified Placeholder)
// Features:
// - Simulates a rising "chaos score" based on unexpected events.
// - This is a conceptual module, representing system instability.
// ======================================================================
module chaos_detector(
    input wire clk,
    input wire reset,
    input wire branch_mispredicted, // Example: Branch misprediction contributes to chaos (from MEM/WB)
    input wire [3:0] mem_access_addr, // Example: Erratic memory access patterns (from MEM)
    input wire [3:0] data_mem_read_data, // Example: Unexpected data values (from MEM)

    output reg [15:0] chaos_score_out // 16-bit output
);

    // Placeholder: Chaos score increases with mispredictions and erratic behavior.
    // In a real system, this would be from complex monitoring.
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            chaos_score_out <= 16'h0000;
        end else begin
            if (branch_mispredicted) begin
                chaos_score_out <= chaos_score_out + 16'h0100; // Significant jump for misprediction
            end

            // Simulate some "erratic" memory access contributing to chaos
            // This is purely illustrative and would need robust detection logic
            // Example: Accessing a forbidden address or unusual data for an address
            if (mem_access_addr == 4'hF && data_mem_read_data == 4'h5) begin // Specific "bad" read pattern
                chaos_score_out <= chaos_score_out + 16'h0050;
            end

            // Gradually decay chaos over time if no new events
            if (chaos_score_out > 16'h0000) begin
                chaos_score_out <= chaos_score_out - 16'h0001;
            end
        end
    end
endmodule


// ======================================================================
// Pattern Detector Module (Conceptual Higher-Order Descriptor Example)
// Enhanced Features:
// - Stores a deeper history of ALU flags using shift registers.
// - Detects MULTIPLE specific "anomalous" patterns across history.
// - Outputs a single "anomaly_detected" flag if ANY pattern matches.
// ======================================================================
module pattern_detector(
    input clk,
    input reset,
    // Current flags represent the flags from the *current* cycle's ALU output (EX stage)
    input wire zero_flag_current,
    input wire negative_flag_current,
    input wire carry_flag_current,
    input wire overflow_flag_current,

    output reg anomaly_detected_out // Output a 1-bit anomaly flag (renamed to match AHO)
);

    // History depth: We'll store current and previous 2 cycles for 3-cycle total view
    parameter HISTORY_DEPTH = 3; // For 3 cycles of data (current, prev1, prev2).

    // Shift registers for ALU flags
    reg [HISTORY_DEPTH-1:0] zero_flag_history;
    reg [HISTORY_DEPTH-1:0] negative_flag_history;
    reg [HISTORY_DEPTH-1:0] carry_flag_history;
    reg [HISTORY_DEPTH-1:0] overflow_flag_history;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            zero_flag_history <= 'b0;
            negative_flag_history <= 'b0;
            carry_flag_history <= 'b0;
            overflow_flag_history <= 'b0;
            anomaly_detected_out <= 1'b0;
        end else begin
            // Shift in current flags, pushing older flags out
            zero_flag_history <= {zero_flag_history[HISTORY_DEPTH-2:0], zero_flag_current};
            negative_flag_history <= {negative_flag_history[HISTORY_DEPTH-2:0], negative_flag_current};
            carry_flag_history <= {carry_flag_history[HISTORY_DEPTH-2:0], carry_flag_current};
            overflow_flag_history <= {overflow_flag_history[HISTORY_DEPTH-2:0], overflow_flag_current};

            // Define Multiple Anomalous Patterns (using current, prev1, prev2 flags)
            // Access: {flag_history[0]} is current, {flag_history[1]} is prev1, {flag_history[2]} is prev2

            wire pattern1_match;
            wire pattern2_match;

            // Pattern 1: (Prev2 Zero=0, Prev1 Negative=1, Current Carry=1)
            // A pattern that might indicate a specific arithmetic flow leading to a problem
            pattern1_match = (!zero_flag_history[2]) && (negative_flag_history[1]) && (carry_flag_history[0]);

            // Pattern 2: (Prev2 Carry=1, Prev1 Overflow=0, Current Zero=0)
            // A pattern that might indicate an unexpected sequence of flags related to overflow/zero conditions
            pattern2_match = (carry_flag_history[2]) && (!overflow_flag_history[1]) && (!zero_flag_history[0]);

            // If ANY defined pattern matches, assert anomaly_detected
            anomaly_detected_out <= pattern1_match || pattern2_match;
        end
    end
endmodule

// ======================================================================
// FSM Entropy Overlay Module
// Description: This module implements a Finite State Machine (FSM) that
//              acts as an overlay, dynamically adjusting system behavior
//              (e.g., stalling, flushing, or locking) based on a
//              combination of machine learning (ML) predictions,
//              internal entropy scores, hazard flags, and various
//              override signals. It's designed to manage system stability
//              and security in the face of unpredictable or anomalous
//              conditions.
//
// States:
// - STATE_OK: Normal operation.
// - STATE_STALL: Halts execution to prevent potential issues, allowing
//                for resolution or re-evaluation.
// - STATE_FLUSH: Clears pipelines or buffers, typically in response to
//                detected corruption or irrecoverable states.
// - STATE_LOCK: Enters a secure, unchangeable state, usually indicating
//               a critical security breach or system integrity compromise.
//
// Inputs:
// - ML Predicted Action: Direct guidance from an ML model on desired
//                        system state.
// - Internal Entropy Score: A measure of randomness or unpredictability
//                           within the system, indicating potential
//                           anomalies or attacks.
// - Internal Hazard Flag: Indicates an architectural hazard within the
//                         system (e.g., data dependency, control hazard).
// - Analog Overrides: External, high-priority signals for immediate
//                     system state changes (lock or flush).
// - Classified Entropy Level: Pre-classified severity of entropy (low, mid, critical).
// - Quantum Override Signal: A highly critical override, possibly from
//                            a quantum-level monitoring system, forcing a lock.
// - Instruction Type: Categorization of the currently executing instruction,
//                     used for context-aware state transitions.
//
// Outputs:
// - FSM State: The current operational state of the FSM.
// - Entropy Log Out: Logs the entropy score when a state transition occurs.
// - Instruction Type Log Out: Logs the instruction type when a state transition occurs.
// ======================================================================
module fsm_entropy_overlay(
    input wire clk,                         // Clock signal
    input wire rst_n,                       // Asynchronous active-low reset
    input wire [1:0] ml_predicted_action,   // Machine Learning model's predicted action
    input wire [7:0] internal_entropy_score,// Current internal entropy score
    input wire internal_hazard_flag,        // Flag indicating an internal system hazard
    input wire analog_lock_override,        // External override to force LOCK state
    input wire analog_flush_override,       // External override to force FLUSH state
    input wire [1:0] classified_entropy_level, // Pre-classified entropy level (Low, Mid, Critical)
    input wire quantum_override_signal,     // Critical override from quantum monitoring
    input wire [2:0] instr_type,            // Type of the current instruction
    output reg [1:0] fsm_state,             // Current FSM state
    output reg [7:0] entropy_log_out,       // Log of entropy score at state change
    output reg [2:0] instr_type_log_out     // Log of instruction type at state change
);

    // --- FSM State Definitions ---
    // These parameters define the possible states of the Finite State Machine.
    parameter STATE_OK    = 2'b00; // Normal operational state
    parameter STATE_STALL = 2'b01; // System stalled, awaiting resolution or re-evaluation
    parameter STATE_FLUSH = 2'b10; // System flushing pipelines/buffers
    parameter STATE_LOCK  = 2'b11; // System locked due to critical event

    // --- ML Action Code Definitions ---
    // These parameters define the actions suggested by the Machine Learning model.
    parameter ML_OK    = 2'b00; // ML suggests normal operation
    parameter ML_STALL = 2'b01; // ML suggests stalling the system
    parameter ML_FLUSH = 2'b10; // ML suggests flushing the system
    parameter ML_LOCK  = 2'b11; // ML suggests locking the system

    // --- Entropy Classification Levels ---
    // These parameters categorize the internal entropy score into predefined levels.
    parameter ENTROPY_LOW      = 2'b00; // Low entropy, normal
    parameter ENTROPY_MID      = 2'b01; // Medium entropy, potentially concerning
    parameter ENTROPY_CRITICAL = 2'b10; // Critical entropy, highly concerning

    // --- Instruction Type Definitions ---
    // These parameters define different categories of instructions processed by the system.
    parameter INSTR_TYPE_ALU    = 3'b000; // Arithmetic Logic Unit operation
    parameter INSTR_TYPE_LOAD   = 3'b001; // Memory load operation
    parameter INSTR_TYPE_STORE  = 3'b010; // Memory store operation
    parameter INSTR_TYPE_BRANCH = 3'b011; // Program control branch instruction
    parameter INSTR_TYPE_JUMP   = 3'b100; // Program control jump instruction

    // --- Thresholds ---
    // Defines the threshold for what is considered a high entropy score.
    parameter ENTROPY_HIGH_THRESHOLD = 8'd180; // Example threshold for high entropy (decimal 180)

    // --- Internal FSM State Registers ---
    // Registers to hold the current and next state of the FSM for synchronous updates.
    reg [1:0] current_state, next_state;

    // ==================================================================
    // Synchronous State Register Logic
    //
    // This always block updates the current FSM state and logs relevant
    // information (entropy score, instruction type) on state transitions.
    // It is sensitive to the positive edge of the clock and the negative
    // edge of the asynchronous reset.
    // ==================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) {
            // Asynchronous reset condition:
            // Reset the FSM to the initial 'OK' state and clear log outputs.
            current_state <= STATE_OK;
            entropy_log_out <= 8'h00;
            instr_type_log_out <= 3'b000;
        } else {
            // Synchronous update:
            // Update the current state to the calculated next state.
            current_state <= next_state;

            // Log entropy and instruction type only when a state transition occurs.
            if (next_state != current_state) {
                entropy_log_out <= internal_entropy_score;
                instr_type_log_out <= instr_type;
            } else {
                // If no state change, clear log outputs (or maintain previous, depending on logging policy).
                entropy_log_out <= 8'h00;
                instr_type_log_out <= 3'b000;
            }
        }
    end

    // ==================================================================
    // Combinational Next-State Logic
    //
    // This always block determines the next FSM state based on current inputs.
    // It's a combinational block, meaning its outputs react immediately
    // to changes in its inputs. Priority is given to override signals.
    // ==================================================================
    always @(*) begin
        // Default to staying in the current state unless a condition dictates a change.
        next_state = current_state;

        // --- High-Priority Overrides ---
        // Quantum override takes highest priority, forcing a LOCK.
        if (quantum_override_signal) begin
            next_state = STATE_LOCK;
        end
        // Analog lock override takes next priority.
        else if (analog_lock_override) begin
            next_state = STATE_LOCK;
        end
        // Analog flush override takes next priority.
        else if (analog_flush_override) begin
            next_state = STATE_FLUSH;
        end
        // --- Normal State Transition Logic ---
        else begin
            // Evaluate based on classified entropy level.
            case (classified_entropy_level)
                ENTROPY_CRITICAL: begin
                    // If entropy is critical, respond aggressively based on instruction type.
                    case (instr_type)
                        INSTR_TYPE_BRANCH, INSTR_TYPE_JUMP:
                            // Branch/Jump with critical entropy suggests a high risk, thus STALL.
                            next_state = STATE_STALL;
                        INSTR_TYPE_LOAD, INSTR_TYPE_STORE:
                            // Load/Store with critical entropy might indicate memory corruption, thus FLUSH.
                            next_state = STATE_FLUSH;
                        INSTR_TYPE_ALU: begin
                            // ALU operations with critical entropy and a hazard flag lead to STALL.
                            if (internal_hazard_flag)
                                next_state = STATE_STALL;
                        end
                        default:
                            // Any other instruction type with critical entropy results in FLUSH.
                            next_state = STATE_FLUSH;
                    endcase
                end
                ENTROPY_MID: begin
                    // If entropy is medium, respond more conservatively based on instruction type
                    // and current state to avoid unnecessary interruptions.
                    case (instr_type)
                        INSTR_TYPE_BRANCH, INSTR_TYPE_JUMP:
                            // Only stall branches/jumps if currently in OK state.
                            if (current_state == STATE_OK)
                                next_state = STATE_STALL;
                        INSTR_TYPE_LOAD, INSTR_TYPE_STORE:
                            // Stall loads/stores if currently in OK or STALL state.
                            if (current_state == STATE_OK || current_state == STATE_STALL)
                                next_state = STATE_STALL;
                        INSTR_TYPE_ALU:
                            // Stall ALU ops if a hazard flag is present.
                            if (internal_hazard_flag)
                                next_state = STATE_STALL;
                        default:
                            // For other types, only stall if there's a hazard and in OK state.
                            if (internal_hazard_flag && current_state == STATE_OK)
                                next_state = STATE_STALL;
                    endcase
                end
                default: begin // ENTROPY_LOW or unclassified
                    // If entropy is low or uncategorized, primary decisions are based on ML predictions
                    // and other internal flags.
                    case (current_state)
                        STATE_OK: begin
                            // In OK state, process ML predictions first.
                            case (ml_predicted_action)
                                ML_STALL: next_state = STATE_STALL;
                                ML_FLUSH: next_state = STATE_FLUSH;
                                ML_LOCK:  next_state = STATE_LOCK;
                                default: begin
                                    // If ML is OK, check internal entropy and hazards.
                                    if (ml_predicted_action == ML_OK &&
                                        internal_entropy_score > ENTROPY_HIGH_THRESHOLD)
                                        next_state = STATE_STALL; // High entropy without ML intervention -> STALL
                                    else if (internal_hazard_flag)
                                        next_state = STATE_STALL; // Internal hazard -> STALL
                                end
                            endcase
                        end
                        STATE_STALL: begin
                            // In STALL state, ML can transition to FLUSH or LOCK, or return to OK if conditions clear.
                            case (ml_predicted_action)
                                ML_FLUSH: next_state = STATE_FLUSH;
                                ML_LOCK:  next_state = STATE_LOCK;
                                default:
                                    // Return to OK if ML is OK, no hazard, and entropy is low.
                                    if (ml_predicted_action == ML_OK &&
                                        !internal_hazard_flag &&
                                        internal_entropy_score <= ENTROPY_HIGH_THRESHOLD)
                                        next_state = STATE_OK;
                            endcase
                        end
                        STATE_FLUSH: begin
                            // In FLUSH state, ML can transition to LOCK, or return to OK/STALL if conditions clear.
                            case (ml_predicted_action)
                                ML_LOCK: next_state = STATE_LOCK;
                                default: begin
                                    // Return to OK if ML is OK, no hazard, and entropy is low.
                                    if (ml_predicted_action == ML_OK &&
                                        !internal_hazard_flag &&
                                        internal_entropy_score <= ENTROPY_HIGH_THRESHOLD)
                                        next_state = STATE_OK;
                                    // If ML suggests STALL, transition to STALL.
                                    else if (ml_predicted_action == ML_STALL)
                                        next_state = STATE_STALL;
                                end
                            endcase
                        end
                        STATE_LOCK:
                            // Once in LOCK state, remain locked until an external reset.
                            next_state = STATE_LOCK;
                        default:
                            // Safety default: if in an undefined state, return to OK.
                            next_state = STATE_OK;
                    endcase
                end
            endcase
        end
    end

    // ==================================================================
    // Output State Assignment
    //
    // This always block continuously assigns the current FSM state to the
    // output register `fsm_state`. This ensures that the external world
    // always sees the current operational state of the module.
    // ==================================================================
    always @(*) begin
        fsm_state = current_state;
    end

endmodule

// ===============================================================================
// ARCHON HAZARD OVERRIDE UNIT (AHO) - Integrated and Enhanced
// Purpose: This module implements the Archon Hazard Override (AHO) unit,
//          responsible for detecting hazardous internal states and generating
//          override signals (flush, stall) for the CPU pipeline.
//
// Key Enhancements:
// 1. Direct incorporation of 'cache_miss_rate_tracker' as a primary input.
// 2. Implementation of 'fluctuating impact' for various metrics through
//    dynamic weighting, controlled by an external 'ml_predicted_action'.
// 3. A sophisticated rule-based decision engine for hazard mitigation,
//    combining dynamically weighted scores with fixed-priority anomaly detection.
// This version is designed to provide 'override_flush_sig' and 'override_stall_sig'
// to the Probabilistic Hazard FSM, rather than direct pipeline control.
// ===============================================================================

module archon_hazard_override_unit (
    input logic                 clk,
    input logic                 rst_n, // Active low reset

    // Core Hazard Metrics (now adapted to 8-bit where needed, Chaos is 16-bit)
    input logic [7:0]           internal_entropy_score_val, // From QED (Quantum Entropy Detector) - 8-bit
    input logic [15:0]          chaos_score_val,            // From CD (Chaos Detector) - 16-bit
    input logic                 anomaly_detected_val,       // From Pattern Detector (high, fixed impact)

    // Performance/System Health Metrics (now adapted to 8-bit where needed)
    input logic [7:0]           branch_miss_rate_tracker,   // Current branch miss rate (from BTB or PMU) - 8-bit
    input logic [7:0]           cache_miss_rate_tracker,    // NEW: Current cache miss rate (from Data Memory/Cache) - 8-bit
    input logic [7:0]           exec_pressure_tracker,      // Current execution pressure (e.g., pipeline fullness) - 8-bit

    // Input from external ML model for dynamic weighting/context
    // This input dictates the current 'risk posture' or 'mode' for hazard detection.
    // Examples: 2'b00=Normal, 2'b01=MonitorRisk, 2'b10=HighRisk, 2'b11=CriticalRisk
    input logic [1:0]           ml_predicted_action,

    // Dynamically scaled thresholds for the combined hazard score (adjusted for new total score range)
    // These thresholds would typically be provided by an external control unit or derived
    // from system-wide context/ML predictions, scaled appropriately for 'total_combined_hazard_score'.
    input logic [20:0]          scaled_flush_threshold,     // If combined score > this, consider flush
    input logic [20:0]          scaled_stall_threshold,     // If combined score > this, consider stall

    // Outputs to CPU pipeline control (specifically for Probabilistic Hazard FSM or main control)
    output logic                override_flush_sig,         // Request for CPU pipeline flush
    output logic                override_stall_sig,         // Request for CPU pipeline stall
    output logic [1:0]          hazard_detected_level       // Severity: 00=None, 01=Low, 10=Medium, 11=High/Critical
);

    // --- Internal Signals for Dynamic Weight Assignment (Fluctuating Impact) ---
    // These 4-bit weights (0-15) are dynamically adjusted based on 'ml_predicted_action'.
    // They amplify or de-emphasize the impact of each raw metric on the total hazard score.
    logic [3:0] W_entropy;
    logic [3:0] W_chaos;
    logic [3:0] W_branch;
    logic [3:0] W_cache;
    logic [3:0] W_exec;

    // --- Internal Signals for Weighted Scores ---
    // Individual weighted scores are calculated by multiplying raw scores by weights.
    // Max product for 8-bit * 4-bit: 255 * 15 = 3825. A 12-bit register is sufficient.
    // Max product for 16-bit * 4-bit: 65535 * 15 = 983025. A 20-bit register is sufficient.
    logic [11:0] weighted_entropy_score;   // 8-bit val * 4-bit weight -> 12-bit
    logic [19:0] weighted_chaos_score;     // 16-bit val * 4-bit weight -> 20-bit
    logic [11:0] weighted_branch_miss_score; // 8-bit val * 4-bit weight -> 12-bit
    logic [11:0] weighted_cache_miss_score;  // 8-bit val * 4-bit weight -> 12-bit
    logic [11:0] weighted_exec_pressure_score; // 8-bit val * 4-bit weight -> 12-bit

    // --- Total Combined Hazard Score ---
    // Sum of all weighted scores.
    // Max sum: (3 * 3825) + (2 * 983025) = 11475 + 1966050 = 1977525.
    // A 21-bit register is sufficient (max value 2097151).
    logic [20:0] total_combined_hazard_score; // Adjusted to 21-bit

    // --- Output Registers (for synchronous outputs) ---
    reg reg_override_flush_sig;
    reg reg_override_stall_sig;
    reg [1:0] reg_hazard_detected_level;

    // --- Clocked Logic for Output Registers ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_override_flush_sig      <= 1'b0;
            reg_override_stall_sig      <= 1'b0;
            reg_hazard_detected_level <= 2'b00; // No hazard detected by default
        end else begin
            // Update output registers with combinational logic's current state
            reg_override_flush_sig      <= override_flush_sig;
            reg_override_stall_sig      <= override_stall_sig;
            reg_hazard_detected_level <= hazard_detected_level;
        end
    end

    // --- Combinational Logic for Dynamic Weight Assignment (Fluctuating Impact) ---
    // This block determines the importance (weights) of each metric based on the
    // 'ml_predicted_action', allowing the system to adapt its sensitivity.
    always @(*) begin
        case (ml_predicted_action)
            2'b00: begin // Normal Operation: Balanced weights, general monitoring
                W_entropy = 4'd8;   // Moderate impact for entropy/chaos
                W_chaos   = 4'd7;
                W_branch  = 4'd5;   // Moderate for branch/cache misses (performance indicators)
                W_cache   = 4'd6;
                W_exec    = 4'd4;   // Lower for execution pressure
            end
            2'b01: begin // Monitor Risk: Increased focus on anomaly/chaos indicators
                W_entropy = 4'd10; // Higher impact for entropy/chaos
                W_chaos   = 4'd9;
                W_branch  = 4'd7;   // Slightly increased for branch/cache misses
                W_cache   = 4'd8;
                W_exec    = 4'd3;   // Reduced emphasis on exec pressure
            end
            2'b10: begin // High Risk: Strong emphasis on potential security/stability issues
                W_entropy = 4'd12; // Significantly higher impact for entropy/chaos
                W_chaos   = 4'd11;
                W_branch  = 4'd9;   // Substantially increased for branch/cache misses (could indicate attack)
                W_cache   = 4'd10;
                W_exec    = 4'd2;   // Minimal emphasis on general performance for immediate risk
            end
            2'b11: begin // Critical Risk: Maximum sensitivity for all hazard indicators
                W_entropy = 4'd15; // Max impact
                W_chaos   = 4'd15; // Max impact
                W_branch  = 4'd13; // Very high impact
                W_cache    = 4'd14; // Very high impact
                W_exec    = 4'd1;   // Almost no impact for exec pressure, focus is on stopping threat
            end
            default: begin // Defensive default: Fallback to normal operation weights
                W_entropy = 4'd8; W_chaos = 4'd7; W_branch = 4'd5; W_cache = 4'd6; W_exec = 4'd4;
            end
        endcase
    end

    // --- Combinational Logic for Weighted Score Calculation (Dynamic Weighted Sum) ---
    // Each raw score is multiplied by its dynamically determined weight.
    assign weighted_entropy_score   = internal_entropy_score_val * W_entropy;
    assign weighted_chaos_score     = chaos_score_val * W_chaos;
    assign weighted_branch_miss_score = branch_miss_rate_tracker * W_branch;
    assign weighted_cache_miss_score    = cache_miss_rate_tracker * W_cache; // NEW: Cache miss included
    assign weighted_exec_pressure_score = exec_pressure_tracker * W_exec;

    // The total combined hazard score aggregates all weighted metric impacts.
    assign total_combined_hazard_score =
        weighted_entropy_score +
        weighted_chaos_score +
        weighted_branch_miss_score +
        weighted_cache_miss_score +
        weighted_exec_pressure_score;

    // --- Combinational Logic for Override Signals (Multi-dimensional Rule Engine) ---
    // This block implements the decision logic, prioritizing different hazard indicators.
    always @(*) begin
        override_flush_sig = 1'b0;
        override_stall_sig = 1'b0;
        hazard_detected_level = 2'b00; // Default to no hazard

        // Rule 1: High-priority anomaly detection (Pattern Detector)
        // If an anomaly is detected, this should trigger a flush immediately,
        // regardless of the combined hazard score, as it signifies a critical state.
        if (anomaly_detected_val) begin
            override_flush_sig = 1'b1;
            hazard_detected_level = 2'b11; // Critical
        end else begin
            // Rule 2: Evaluate based on combined hazard score against dynamic thresholds
            if (total_combined_hazard_score > scaled_flush_threshold) begin
                override_flush_sig = 1'b1;
                hazard_detected_level = 2'b10; // Medium to High (depending on threshold severity)
            end else if (total_combined_hazard_score > scaled_stall_threshold) begin
                override_stall_sig = 1'b1;
                hazard_detected_level = 2'b01; // Low to Medium
            end else begin
                // No significant hazard detected by AHO's scoring system
                override_flush_sig = 1'b0;
                override_stall_sig = 1'b0;
                hazard_detected_level = 2'b00; // None
            end
        end
    end

    // Outputs are registered, so assign the internal registered signals
    // These outputs directly drive the next stage (the new FSM)
    // No need for separate output assigns here since they are declared as logic within the module
    // and directly assigned in the always_comb block and then registered.
    // Remove the previous 'assign override_flush_sig_out = reg_override_flush_sig;' style lines.
endmodule

// ======================================================================
// NEW: Entropy Control Logic Module
// Features:
// - Directly uses the 16-bit external entropy input from 'entropy_bus.txt'.
// - Applies simple, configurable thresholds to generate stall/flush signals.
// - This module provides the *base* entropy-driven control signals.
//   These can then be modulated by ML and chaos predictors in the main CPU.
// ===============================================================================
module entropy_control_logic(
    input wire [15:0] external_entropy_in, // 16-bit external entropy from entropy_bus.txt
    output wire entropy_stall,            // Assert to signal a basic entropy-induced stall
    output wire entropy_flush             // Assert to signal a basic entropy-induced flush
);

    // Define entropy thresholds for stall and flush
    // These values are for a 16-bit (0-65535) entropy input.
    parameter ENTROPY_STALL_THRESHOLD = 16'd10000;  // Example: Below 10000, consider stalling
    parameter ENTROPY_FLUSH_THRESHOLD = 16'd50000; // Example: Above 50000, consider flushing

    assign entropy_stall = (external_entropy_in < ENTROPY_STALL_THRESHOLD);
    assign entropy_flush = (external_entropy_in > ENTROPY_FLUSH_THRESHOLD);

endmodule


// ===============================================================================
// Pipeline CPU Core (INTEGRATED VERSION)
// Combines all previously defined modules and orchestrates their interactions.
// ===============================================================================
module pipeline_cpu(
    input wire clk,
    input wire reset, // Active high reset (converts to active low for some modules)
    input wire [15:0] external_entropy_in, // Input from entropy_bus.txt (for Entropy Control Logic)
    input wire [1:0] ml_predicted_action, // ML model's predicted action for AHO and FSM

    // START OF ADDED PARTS: Analog Override Inputs for pipeline_cpu and Quantum Override
    input wire analog_lock_override_in,  // From top-level analog controller
    input wire analog_flush_override_in, // From top-level analog controller
    input wire quantum_override_signal_in, // NEW: Quantum override signal from Qiskit simulation
    // END OF ADDED PARTS

    output wire [3:0] debug_pc,         // For debugging: current PC
    output wire [15:0] debug_instr,     // For debugging: current instruction
    output wire debug_stall,            // For debugging: indicates pipeline stall
    output wire debug_flush,            // For debugging: indicates pipeline flush
    output wire debug_lock,             // For debugging: indicates system lock
    output wire [7:0] debug_fsm_entropy_log, // For debugging: entropy value logged by new FSM
    output wire [2:0] debug_fsm_instr_type_log // NEW: Debug output for logged instruction type
);

    // --- Active Low Reset for Modules that use it ---
    wire rst_n = ~reset;

    // --- Internal Wires & Registers for Pipeline Stages ---
    // IF Stage
    reg [3:0] pc_reg;
    wire [15:0] if_instr; // Instruction fetched
    wire [3:0] if_pc_plus_1; // Changed to +1 as PC is 4-bit, not byte-addressed
    wire [3:0] next_pc; // The next PC to load into pc_reg

    // IF/ID Pipeline Register
    reg [3:0] if_id_pc_plus_1_reg; // For branch target calc and next PC
    reg [15:0] if_id_instr_reg;    // Instruction for ID stage

    // ID Stage
    wire [3:0] id_pc_plus_1;
    wire [15:0] id_instr;
    wire [3:0] id_operand1;
    wire [3:0] id_operand2;
    wire [2:0] id_rs1_addr;
    wire [2:0] id_rs2_addr;
    wire [2:0] id_rd_addr;
    wire [2:0] id_alu_op;        // Decoded ALU operation
    wire [3:0] id_immediate;     // Sign-extended immediate value (simplified 3-bit imm to 4-bit)
    wire id_reg_write_enable;    // Write enable for RegFile
    wire id_mem_read_enable;     // Read enable for Data Memory
    wire id_mem_write_enable;    // Write enable for Data Memory
    wire id_is_branch_inst;      // Decoded as a branch instruction
    wire id_is_jump_inst;        // Decoded as a jump instruction
    wire [3:0] id_branch_target; // Branch target from instruction (simplified 3-bit to 4-bit)

    // START OF ADDED PARTS: Wire for mapped instruction type
    wire [2:0] instr_type_to_fsm_wire;
    // END OF ADDED PARTS

    // ID/EX Pipeline Register
    reg [3:0] id_ex_pc_plus_1_reg;
    reg [3:0] id_ex_operand1_reg;
    reg [3:0] id_ex_operand2_reg;
    reg [2:0] id_ex_rd_addr_reg;
    reg [2:0] id_ex_alu_op_reg;
    reg id_ex_reg_write_enable_reg;
    reg id_ex_mem_read_enable_reg;
    reg id_ex_mem_write_enable_reg;
    reg id_ex_is_branch_inst_reg;
    reg id_ex_is_jump_inst_reg;
    reg [3:0] id_ex_branch_target_reg;
    reg [15:0] id_ex_instr_reg; // For Quantum Entropy Detector

    // EX Stage
    wire [3:0] ex_alu_operand1; // Could be forwarded value
    wire [3:0] ex_alu_operand2; // Could be forwarded value (for ALU computation or mem_write_data)
    wire [3:0] ex_alu_result;
    wire ex_zero_flag;
    wire ex_negative_flag;
    wire ex_carry_flag;
    wire ex_overflow_flag;
    wire [2:0] ex_rd_addr;
    wire ex_reg_write_enable;
    wire ex_mem_read_enable;
    wire ex_mem_write_enable;
    wire ex_is_branch_inst;
    wire ex_is_jump_inst;
    wire [3:0] ex_branch_target; // Target for actual branch
    wire [3:0] ex_branch_pc;     // PC of the branch instruction itself for misprediction check

    // EX/MEM Pipeline Register
    reg [3:0] ex_mem_alu_result_reg;
    reg [3:0] ex_mem_mem_write_data_reg; // Value to write to Data Memory
    reg [2:0] ex_mem_rd_addr_reg;
    reg ex_mem_reg_write_enable_reg;
    reg ex_mem_mem_read_enable_reg;
    reg ex_mem_mem_write_enable_reg;
    reg ex_mem_zero_flag_reg;      // ALU zero flag for branch resolution
    reg ex_mem_is_branch_inst_reg; // Branch instruction flag
    reg ex_mem_is_jump_inst_reg;   // Jump instruction flag
    reg [3:0] ex_mem_pc_plus_1_reg; // PC + 1 from IF stage
    reg [3:0] ex_mem_branch_target_reg; // Branch target from instruction
    reg [3:0] ex_mem_branch_pc_reg; // PC of the branch instruction in EX stage

    // MEM Stage
    wire [3:0] mem_read_data;     // Data read from Data Memory
    wire [3:0] mem_alu_result;    // ALU result passed from EX/MEM
    wire [2:0] mem_rd_addr;       // Destination register address
    wire mem_reg_write_enable;    // RegFile write enable
    wire mem_mem_read_enable;     // Data Memory read enable
    wire mem_mem_write_enable;    // Data Memory write enable
    wire [3:0] mem_mem_addr;      // Address for Data Memory (ALU result)

    wire branch_actual_taken;     // Actual outcome of branch
    wire branch_mispredicted;     // Signal to BTB and Chaos Detector
    wire [3:0] branch_resolved_pc; // PC of resolved branch (from EX/MEM pc_reg)
    wire [3:0] branch_resolved_target_pc; // Actual target of resolved branch

    // MEM/WB Pipeline Register
    reg [3:0] mem_wb_write_data_reg; // Data to write to RegFile (ALU result or MemRead data)
    reg [2:0] mem_wb_rd_addr_reg;    // Destination register address
    reg mem_wb_reg_write_enable_reg; // RegFile write enable

    // WB Stage
    wire [3:0] wb_write_data;     // Final data for RegFile write
    wire [2:0] wb_rd_addr;        // Final destination register
    wire wb_reg_write_enable;     // Final RegFile write enable

    // --- Pipeline Control Signals ---
    wire pipeline_stall; // Overall stall signal
    wire pipeline_flush; // Overall flush signal

    // For simplicity, tracking rough execution pressure: number of active instructions
    reg [7:0] exec_pressure_counter; // Example: count of non-NOPs in flight (simplified)
    reg [7:0] cache_miss_rate_dummy; // Placeholder for actual cache miss rate. Assume 0-255 scaling.

    // AHO internal hazard signals (outputs from AHO)
    wire aho_override_flush_req;
    wire aho_override_stall_req;
    wire [1:0] aho_hazard_level;

    // Consolidated internal hazard flag for the new FSM
    wire new_fsm_internal_hazard_flag;
    wire [1:0] new_fsm_control_signal; // Output from the new entropy-aware FSM
    wire [7:0] new_fsm_entropy_log;    // Entropy log from the new FSM
    wire [2:0] new_fsm_instr_type_log; // NEW: Instruction type log from the new FSM

    // Dummy values for AHO thresholds (these would come from ML inference)
    // Updated to match the 21-bit total_combined_hazard_score
    localparam AHO_SCALED_FLUSH_THRESH = 21'd1000000; // Example: approx halfway of max score
    localparam AHO_SCALED_STALL_THRESH = 21'd500000;  // Example: approx quarter of max score

    // START OF ADDED PARTS: Wire for classified entropy level
    wire [1:0] classified_entropy_level_wire;
    // END OF ADDED PARTS

    // --- Instantiate Sub-modules ---

    // Instruction Memory
    instruction_ram i_imem (
        .clk(clk),
        .reset(reset),
        .pc_in(pc_reg),
        .instr_opcode(if_instr)
    );

    // Register File
    register_file i_regfile (
        .clk(clk),
        .reset(reset),
        .regfile_write_enable(wb_reg_write_enable),
        .write_addr(wb_rd_addr),
        .write_data(wb_write_data),
        .read_addr1(id_rs1_addr),
        .read_addr2(id_rs2_addr),
        .read_data1(id_operand1),
        .read_data2(id_operand2)
    );

    // ALU Unit
    alu_unit i_alu (
        .alu_operand1(ex_alu_operand1),
        .alu_operand2(ex_alu_operand2),
        .alu_op(id_ex_alu_op_reg),
        .alu_result(ex_alu_result),
        .zero_flag(ex_zero_flag),
        .negative_flag(ex_negative_flag),
        .carry_flag(ex_carry_flag),
        .overflow_flag(ex_overflow_flag)
    );

    // Data Memory
    data_mem i_dmem (
        .clk(clk),
        .mem_write_enable(mem_mem_write_enable),
        .mem_read_enable(mem_mem_read_enable),
        .addr(mem_mem_addr),
        .write_data(ex_mem_mem_write_data_reg),
        .read_data(mem_read_data)
    );

    // Branch Target Buffer
    wire [3:0] if_btb_predicted_next_pc; // From BTB prediction
    wire if_btb_predicted_taken;   // From BTB prediction
    branch_target_buffer i_btb (
        .clk(clk),
        .reset(reset),
        .pc_in(pc_reg), // Current PC to get prediction for
        .branch_resolved_pc(branch_resolved_pc),
        .branch_resolved_pc_valid(ex_mem_is_branch_inst_reg || ex_mem_is_jump_inst_reg), // Valid if it was a branch or jump
        .branch_resolved_target_pc(branch_resolved_target_pc),
        .branch_resolved_taken(branch_actual_taken),
        .predicted_next_pc(if_btb_predicted_next_pc), // Output from BTB for IF
        .predicted_taken(if_btb_predicted_taken)    // Output from BTB for IF
    );

    // Quantum Entropy Detector
    wire [3:0] qed_instr_opcode_input; // Extracted opcode for QED
    assign qed_instr_opcode_input = id_ex_instr_reg[15:12]; // Assuming opcode is 4 MSBs of ID/EX instruction
    wire qed_reset = reset; // QED uses active high reset
    wire [7:0] qed_entropy_score_out; // 8-bit output for AHO and new FSM
    quantum_entropy_detector i_qed (
        .clk(clk),
        .reset(qed_reset),
        .instr_opcode(qed_instr_opcode_input),
        .alu_result(ex_alu_result),
        .zero_flag(ex_zero_flag),
        .entropy_score_out(qed_entropy_score_out)
    );

    // Chaos Detector
    wire cd_reset = reset; // CD uses active high reset
    wire [15:0] cd_chaos_score_out; // 16-bit output for AHO
    chaos_detector i_chaos_detector (
        .clk(clk),
        .reset(cd_reset),
        .branch_mispredicted(branch_mispredicted),
        .mem_access_addr(mem_mem_addr), // Address used in MEM stage
        .data_mem_read_data(mem_read_data), // Data read in MEM stage
        .chaos_score_out(cd_chaos_score_out)
    );

    // Pattern Detector
    wire pd_reset = reset; // PD uses active high reset
    wire pd_anomaly_detected_out; // 1-bit output for AHO
    pattern_detector i_pattern_detector (
        .clk(clk),
        .reset(pd_reset),
        .zero_flag_current(ex_zero_flag),
        .negative_flag_current(ex_negative_flag),
        .carry_flag_current(ex_carry_flag),
        .overflow_flag_current(ex_overflow_flag),
        .anomaly_detected_out(pd_anomaly_detected_out)
    );

    // Archon Hazard Override Unit (AHO)
    archon_hazard_override_unit i_aho (
        .clk                        (clk),
        .rst_n                      (rst_n), // Active low reset

        .internal_entropy_score_val (qed_entropy_score_out), // Now 8-bit
        .chaos_score_val            (cd_chaos_score_out),
        .anomaly_detected_val       (pd_anomaly_detected_out),

        .branch_miss_rate_tracker   (debug_branch_miss_rate), // Use the debug output for now (8-bit)
        .cache_miss_rate_tracker    (cache_miss_rate_dummy),  // Placeholder (needs actual cache logic) (8-bit)
        .exec_pressure_tracker      (exec_pressure_counter),  // Use the simplified counter (8-bit)

        .ml_predicted_action        (ml_predicted_action),    // From top-level input
        .scaled_flush_threshold     (AHO_SCALED_FLUSH_THRESH), // Fixed for this example, or from ML
        .scaled_stall_threshold     (AHO_SCALED_STALL_THRESH), // Fixed for this example, or from ML

        .override_flush_sig         (aho_override_flush_req), // Output to new FSM
        .override_stall_sig         (aho_override_stall_req), // Output to new FSM
        .hazard_detected_level      (aho_hazard_level)        // For debug or other system management
    );

    // Instantiate entropy_trigger_decoder
    entropy_trigger_decoder i_entropy_decoder (
        .entropy_in(qed_entropy_score_out),        // Connect QED output to decoder input
        .signal_class(classified_entropy_level_wire) // Output to new wire
    );

    // START OF ADDED PARTS: Map id_opcode (4-bit) to instr_type (3-bit) for FSM
    wire [3:0] id_opcode_from_id_stage = if_id_instr_reg[15:12]; // Use the opcode from the ID/EX pipeline register
    always @(*) begin
        case (id_opcode_from_id_stage)
            4'h1, 4'h2, 4'h3, 4'h6: instr_type_to_fsm_wire = 3'b000; // ADD, ADDI, SUB, XOR -> ALU
            4'h4:                   instr_type_to_fsm_wire = 3'b001; // LD -> LOAD
            4'h5:                   instr_type_to_fsm_wire = 3'b010; // ST -> STORE
            4'h7:                   instr_type_to_fsm_wire = 3'b011; // BEQ -> BRANCH
            4'h8:                   instr_type_to_fsm_wire = 3'b100; // JUMP -> JUMP
            default:                instr_type_to_fsm_wire = 3'b111; // Reserved/Other
        endcase
    end
    // END OF ADDED PARTS

    // NEW: Entropy-Aware FSM
    // Consolidate AHO's requests into a single internal hazard flag for the new FSM
    assign new_fsm_internal_hazard_flag = aho_override_flush_req || aho_override_stall_req;
    fsm_entropy_overlay i_entropy_fsm (
        .clk(clk),
        .rst_n(rst_n), // Active low reset
        .ml_predicted_action(ml_predicted_action),    // ML model's prediction
        .internal_entropy_score(qed_entropy_score_out), // Entropy score from QED
        .internal_hazard_flag(new_fsm_internal_hazard_flag),
        // START OF ADDED PARTS: Passing analog, quantum, and instruction type inputs to FSM
        .analog_lock_override(analog_lock_override_in),
        .analog_flush_override(analog_flush_override_in),
        .classified_entropy_level(classified_entropy_level_wire), // Pass classified entropy level to FSM
        .quantum_override_signal(quantum_override_signal_in), // Pass quantum override signal to FSM
        .instr_type(instr_type_to_fsm_wire), // NEW: Pass mapped instruction type to FSM
        // END OF ADDED PARTS
        .fsm_state(new_fsm_control_signal),           // Main pipeline control output
        .entropy_log_out(new_fsm_entropy_log),         // Debug output for entropy logging
        .instr_type_log_out(new_fsm_instr_type_log)    // NEW: Debug output for instruction type logging
    );

    // Entropy Control Logic (for external entropy input) - remains as a separate "base" influence
    wire entropy_ctrl_stall;
    wire entropy_ctrl_flush;
    entropy_control_logic i_entropy_ctrl (
        .external_entropy_in(external_entropy_in),
        .entropy_stall(entropy_ctrl_stall),
        .entropy_flush(entropy_ctrl_flush)
    );

    // --- Pipeline Control Unit ---
    // Combines all stall/flush requests, now primarily driven by the new entropy-aware FSM.
    // External entropy control acts as an additional independent trigger for stall/flush.
    // Prioritize LOCK > FLUSH > STALL
    assign pipeline_flush = (new_fsm_control_signal == 2'b10) || // FSM requests FLUSH
                            (new_fsm_control_signal == 2'b11) || // FSM requests LOCK (implies FLUSH)
                            entropy_ctrl_flush;                     // External entropy requests FLUSH

    assign pipeline_stall = (new_fsm_control_signal == 2'b01) || // FSM requests STALL
                            (new_fsm_control_signal == 2'b11) || // FSM requests LOCK (implies STALL)
                            entropy_ctrl_stall;                     // External entropy requests STALL

    // --- Execution Pressure Counter (Simplified) ---
    // Increment if not a NOP, decrement if stall or flush occurs (rough heuristic)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            exec_pressure_counter <= 8'h0;
            cache_miss_rate_dummy <= 8'h0; // Initialize dummy cache miss rate
        end else if (pipeline_flush) begin
            exec_pressure_counter <= 8'h0; // Clear on flush
            cache_miss_rate_dummy <= 8'h0; // Clear dummy cache miss rate on flush
        end else if (pipeline_stall) begin
            // Hold or slightly decrement
            exec_pressure_counter <= exec_pressure_counter;
            cache_miss_rate_dummy <= cache_miss_rate_dummy;
        end else begin
            // Assuming opcode 4'h9 is NOP
            if (if_id_instr_reg[15:12] != 4'h9) begin // If instruction is not NOP
                if (exec_pressure_counter < 8'hFF)
                    exec_pressure_counter <= exec_pressure_counter + 8'h1;
            end else {
                if (exec_pressure_counter > 8'h0)
                    exec_pressure_counter <= exec_pressure_counter - 8'h1;
            }
            // Simulate a dummy cache miss rate that fluctuates
            if ($urandom_range(0, 100) < 5) begin // 5% chance to increase
                if (cache_miss_rate_dummy < 8'hFF)
                    cache_miss_rate_dummy <= cache_miss_rate_dummy + 8'h1;
            end else if ($urandom_range(0, 100) < 10) begin // 10% chance to decrease
                if (cache_miss_rate_dummy > 8'h0)
                    cache_miss_rate_dummy <= cache_miss_rate_dummy - 8'h1;
            }
        }
    end

    // --- IF Stage (Instruction Fetch) ---
    // PC calculation and instruction fetch
    assign if_pc_plus_1 = pc_reg + 4'b0001; // Assuming PC increments by 1 per instruction

    // Next PC logic, considering branches, jumps, and pipeline hazards
    always @(*) begin
        next_pc = if_pc_plus_1; // Default: increment PC

        // Branch/Jump override
        if (ex_mem_is_jump_inst_reg) begin // Resolved Jump
            next_pc = ex_mem_branch_target_reg;
        } else if (ex_mem_is_branch_inst_reg) begin // Resolved Branch
            if (branch_actual_taken) begin
                next_pc = ex_mem_branch_target_reg;
            } else { // If not taken or mispredicted (predicted taken but actually not taken)
                next_pc = ex_mem_pc_plus_1_reg; // Not taken, use PC+1 from EX stage
            }
        } else if (if_btb_predicted_taken) begin // BTB Prediction
            next_pc = if_btb_predicted_next_pc;
        }

        // Hazard overrides (new FSM has highest priority for pipeline control)
        if (new_fsm_control_signal == 2'b11) begin // LOCK state
            next_pc = 4'h0; // Force PC to 0 on lock
        end else if (new_fsm_control_signal == 2'b10) begin // FLUSH state
            next_pc = 4'h0; // Flush: reset PC to 0 or entry point
        end else if (new_fsm_control_signal == 2'b01) begin // STALL state
            next_pc = pc_reg; // Stall: keep current PC, refetch same instruction
        }
        // If new_fsm_control_signal is STATE_OK (2'b00), no override, so normal PC flow
    end


    // PC Register Update
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pc_reg <= 4'h0;
        } else begin
            pc_reg <= next_pc;
        end
    end

    // IF/ID Pipeline Register
    always @(posedge clk or posedge reset) begin
        if (reset || pipeline_flush) begin // Flush clears pipeline registers
            if_id_pc_plus_1_reg <= 4'h0;
            if_id_instr_reg <= 16'h0000; // NOP
        } else if (~pipeline_stall) { // Stall holds pipeline registers
            if_id_pc_plus_1_reg <= if_pc_plus_1;
            if_id_instr_reg <= if_instr;
        }
    end

    // --- ID Stage (Instruction Decode / Register Fetch) ---
    assign id_pc_plus_1 = if_id_pc_plus_1_reg;
    assign id_instr = if_id_instr_reg;

    // Instruction Decode (simplified)
    // Assume common instruction format: [opcode (4)|rd (3)|rs1 (3)|rs2 (3)|imm (3)]
    // opcodes: 1=ADD, 2=ADDI, 3=SUB, 4=LD, 5=ST, 6=XOR, 7=BEQ, 8=JUMP, 9=NOP
    wire [3:0] id_opcode = id_instr[15:12];
    assign id_rd_addr = id_instr[11:9];
    assign id_rs1_addr = id_instr[8:6];
    assign id_rs2_addr = id_instr[5:3];
    assign id_immediate = {1'b0, id_instr[2:0]}; // Simplified: 3-bit immediate, sign-extended to 4 bits

    // Determine control signals based on opcode (simplified)
    assign id_reg_write_enable = (id_opcode == 4'h1 || id_opcode == 4'h2 || id_opcode == 4'h3 ||
                                  id_opcode == 4'h4 || id_opcode == 4'h6 || id_opcode == 4'h0); // R0 is 0
    assign id_mem_read_enable  = (id_opcode == 4'h4); // LD
    assign id_mem_write_enable = (id_opcode == 4'h5); // ST
    assign id_is_branch_inst   = (id_opcode == 4'h7); // BEQ
    assign id_is_jump_inst     = (id_opcode == 4'h8); // JUMP
    assign id_branch_target    = id_instr[3:0]; // Simplified: 4-bit relative offset/absolute target

    // ALU opcodes (simplified mapping)
    always @(*) begin
        case (id_opcode)
            4'h1: id_alu_op = 3'b000; // ADD
            4'h2: id_alu_op = 3'b000; // ADDI (add immediate)
            4'h3: id_alu_op = 3'b001; // SUB
            4'h4: id_alu_op = 3'b000; // LD (for address calculation)
            4'h5: id_alu_op = 3'b000; // ST (for address calculation)
            4'h6: id_alu_op = 3'b100; // XOR
            4'h7: id_alu_op = 3'b001; // BEQ (for comparison: op1 - op2 == 0)
            default: id_alu_op = 3'bXXX; // Undefined/NOP
        endcase
    end

    // --- Hazard Detection and Forwarding (Simplified Data Hazards) ---
    // Detect RAW hazard between EX/MEM and ID (rs1/rs2)
    wire ex_mem_writes_to_rs1_id = ex_mem_reg_write_enable_reg && (ex_mem_rd_addr_reg == id_rs1_addr);
    wire ex_mem_writes_to_rs2_id = ex_mem_reg_write_enable_reg && (ex_mem_rd_addr_reg == id_rs2_addr);

    // Detect RAW hazard between MEM/WB and ID (rs1/rs2)
    wire mem_wb_writes_to_rs1_id = mem_wb_reg_write_enable_reg && (mem_wb_rd_addr_reg == id_rs1_addr);
    wire mem_wb_writes_to_rs2_id = mem_wb_reg_write_enable_reg && (mem_wb_rd_addr_reg == id_rs2_addr);

    // Forwarding logic (simplified: direct connection if hazard)
    wire [3:0] forward_operand1;
    wire [3:0] forward_operand2;

    assign forward_operand1 = (ex_mem_writes_to_rs1_id && (id_rs1_addr != 3'b000)) ? ex_mem_alu_result_reg :
                              (mem_wb_writes_to_rs1_id && (id_rs1_addr != 3'b000)) ? mem_wb_write_data_reg :
                              id_operand1; // Default to RegFile read

    assign forward_operand2 = (ex_mem_writes_to_rs2_id && (id_rs2_addr != 3'b000)) ? ex_mem_alu_result_reg :
                              (mem_wb_writes_to_rs2_id && (id_rs2_addr != 3'b000)) ? mem_wb_write_data_reg :
                              id_operand2; // Default to RegFile read


    // ID/EX Pipeline Register
    always @(posedge clk or posedge reset) begin
        if (reset || pipeline_flush) begin
            id_ex_pc_plus_1_reg <= 4'h0;
            id_ex_operand1_reg <= 4'h0;
            id_ex_operand2_reg <= 4'h0;
            id_ex_rd_addr_reg <= 3'h0;
            id_ex_alu_op_reg <= 3'h0;
            id_ex_reg_write_enable_reg <= 1'b0;
            id_ex_mem_read_enable_reg <= 1'b0;
            id_ex_mem_write_enable_reg <= 1'b0;
            id_ex_is_branch_inst_reg <= 1'b0;
            id_ex_is_jump_inst_reg <= 1'b0;
            id_ex_branch_target_reg <= 4'h0;
            id_ex_instr_reg <= 16'h0000; // NOP
        } else if (~pipeline_stall) {
            id_ex_pc_plus_1_reg <= id_pc_plus_1;
            // Select operand 2 based on instruction type (immediate or register)
            id_ex_operand1_reg <= forward_operand1;
            id_ex_operand2_reg <= (id_opcode == 4'h2 || id_opcode == 4'h4 || id_opcode == 4'h5) ? id_immediate : forward_operand2;
            id_ex_rd_addr_reg <= id_rd_addr;
            id_ex_alu_op_reg <= id_alu_op;
            id_ex_reg_write_enable_reg <= id_reg_write_enable;
            id_ex_mem_read_enable_reg <= id_mem_read_enable;
            id_ex_mem_write_enable_reg <= id_mem_write_enable;
            id_ex_is_branch_inst_reg <= id_is_branch_inst;
            id_ex_is_jump_inst_reg <= id_is_jump_inst;
            id_ex_branch_target_reg <= id_branch_target;
            id_ex_instr_reg <= id_instr;
        }
    }

    // --- EX Stage (Execute) ---
    assign ex_alu_operand1 = id_ex_operand1_reg;
    assign ex_alu_operand2 = id_ex_operand2_reg; // This is the ALU's second operand or store data
    assign ex_rd_addr = id_ex_rd_addr_reg;
    assign ex_reg_write_enable = id_ex_reg_write_enable_reg;
    assign ex_mem_read_enable = id_ex_mem_read_enable_reg;
    assign ex_mem_write_enable = id_ex_mem_write_enable_reg;
    assign ex_is_branch_inst = id_ex_is_branch_inst_reg;
    assign ex_is_jump_inst = id_ex_is_jump_inst_reg;
    assign ex_branch_target = id_ex_branch_target_reg;
    assign ex_branch_pc = id_ex_pc_plus_1_reg - 4'b0001; // PC of the branch instruction itself

    // Calculate actual branch target: PC of branch instruction + branch offset
    wire [3:0] actual_branch_target_calc;
    assign actual_branch_target_calc = ex_branch_pc + ex_branch_target;

    // EX/MEM Pipeline Register
    always @(posedge clk or posedge reset) begin
        if (reset || pipeline_flush) begin
            ex_mem_alu_result_reg <= 4'h0;
            ex_mem_mem_write_data_reg <= 4'h0;
            ex_mem_rd_addr_reg <= 3'h0;
            ex_mem_reg_write_enable_reg <= 1'b0;
            ex_mem_mem_read_enable_reg <= 1'b0;
            ex_mem_mem_write_enable_reg <= 1'b0;
            ex_mem_zero_flag_reg <= 1'b0;
            ex_mem_is_branch_inst_reg <= 1'b0;
            ex_mem_is_jump_inst_reg <= 1'b0;
            ex_mem_pc_plus_1_reg <= 4'h0;
            ex_mem_branch_target_reg <= 4'h0;
            ex_mem_branch_pc_reg <= 4'h0;
        } else if (~pipeline_stall) {
            ex_mem_alu_result_reg <= ex_alu_result;
            ex_mem_mem_write_data_reg <= ex_alu_operand2; // For ST instructions
            ex_mem_rd_addr_reg <= ex_rd_addr;
            ex_mem_reg_write_enable_reg <= ex_reg_write_enable;
            ex_mem_mem_read_enable_reg <= ex_mem_read_enable;
            ex_mem_mem_write_enable_reg <= ex_mem_write_enable;
            ex_mem_zero_flag_reg <= ex_zero_flag; // Pass zero flag for branch check
            ex_mem_is_branch_inst_reg <= ex_is_branch_inst;
            ex_mem_is_jump_inst_reg <= ex_is_jump_inst;
            ex_mem_pc_plus_1_reg <= id_ex_pc_plus_1_reg; // Pass PC+1 from ID stage
            ex_mem_branch_target_reg <= actual_branch_target_calc; // Actual calculated target
            ex_mem_branch_pc_reg <= ex_branch_pc; // PC of the branch instruction
        }
    end

    // --- MEM Stage (Memory Access) ---
    assign mem_alu_result = ex_mem_alu_result_reg;
    assign mem_rd_addr = ex_mem_rd_addr_reg;
    assign mem_reg_write_enable = ex_mem_reg_write_enable_reg;
    assign mem_mem_read_enable = ex_mem_mem_read_enable_reg;
    assign mem_mem_write_enable = ex_mem_mem_write_enable_reg;
    assign mem_mem_addr = ex_mem_alu_result_reg; // ALU result is memory address

    // Branch Resolution in MEM Stage
    assign branch_actual_taken = ex_mem_is_branch_inst_reg && ex_mem_zero_flag_reg; // BEQ taken if Zero flag is set
    assign branch_resolved_pc = ex_mem_branch_pc_reg;
    assign branch_resolved_target_pc = ex_mem_branch_target_reg;

    // Branch Misprediction Detection
    wire branch_mispredicted_local; // Local wire for branch misprediction
    assign branch_mispredicted = branch_mispredicted_local; // Assign to top-level wire

    always @(*) begin
        branch_mispredicted_local = 1'b0; // Default to no misprediction

        // Only check misprediction if it was a branch/jump instruction that completed EX stage
        if (ex_mem_is_branch_inst_reg || ex_mem_is_jump_inst_reg) begin
            if (ex_mem_is_branch_inst_reg) begin // Conditional branch (BEQ)
                // Misprediction if predicted taken != actual taken
                // OR if predicted target != actual target (if taken)
                if (if_btb_predicted_taken != branch_actual_taken) begin
                    branch_mispredicted_local = 1'b1;
                end else if (branch_actual_taken && (if_btb_predicted_next_pc != branch_resolved_target_pc)) begin
                    branch_mispredicted_local = 1'b1;
                }
            end else if (ex_mem_is_jump_inst_reg) begin // Unconditional jump
                // Misprediction if predicted target != actual target
                if (if_btb_predicted_next_pc != branch_resolved_target_pc) begin
                    branch_mispredicted_local = 1'b1;
                }
            end
        }
    end

    // For debugging branch miss rate
    reg [7:0] branch_miss_rate_counter;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            branch_miss_rate_counter <= 8'h0;
        } else if (branch_mispredicted) begin
            if (branch_miss_rate_counter < 8'hFF)
                branch_miss_rate_counter <= branch_miss_rate_counter + 8'h1;
        } else {
            if (branch_miss_rate_counter > 8'h0)
                branch_miss_rate_counter <= branch_miss_rate_counter - 8'h01; // Decay
        }
    end
    wire [7:0] debug_branch_miss_rate = branch_miss_rate_counter; // Output for AHO and debug

    // MEM/WB Pipeline Register
    always @(posedge clk or posedge reset) begin
        if (reset || pipeline_flush) begin
            mem_wb_write_data_reg <= 4'h0;
            mem_wb_rd_addr_reg <= 3'h0;
            mem_wb_reg_write_enable_reg <= 1'b0;
        } else if (~pipeline_stall) {
            // Data to write back: from memory if Load, else from ALU
            mem_wb_write_data_reg <= (mem_mem_read_enable) ? mem_read_data : mem_alu_result;
            mem_wb_rd_addr_reg <= mem_rd_addr;
            mem_wb_reg_write_enable_reg <= mem_reg_write_enable;
        }
    end

    // --- WB Stage (Write Back) ---
    assign wb_write_data = mem_wb_write_data_reg;
    assign wb_rd_addr = mem_wb_rd_addr_reg;
    assign wb_reg_write_enable = mem_wb_reg_write_enable_reg;

    // --- Debug Outputs ---
    assign debug_pc = pc_reg;
    assign debug_instr = if_instr; // Or if_id_instr_reg, depending on desired debug point
    assign debug_stall = pipeline_stall;
    assign debug_flush = pipeline_flush;
    assign debug_lock = (new_fsm_control_signal == 2'b11); // Directly from new FSM lock state
    assign debug_fsm_entropy_log = new_fsm_entropy_log; // New debug output for entropy logging
    assign debug_fsm_instr_type_log = new_fsm_instr_type_log; // NEW: Debug output for logged instruction type

endmodule


// ===============================================================================
// NEW MODULE: entropy_trigger_decoder.v
// Purpose: Simulates compression of incoming analog entropy signals (8-bit)
//          into meaningful trigger vectors or score levels (2-bit).
// ===============================================================================
module entropy_trigger_decoder(
    input wire [7:0] entropy_in,    // 8-bit entropy score (0-255)
    output reg [1:0] signal_class   // 2-bit output: 00 = LOW, 01 = MID, 10 = CRITICAL
);

    // Define thresholds for classification
    parameter THRESHOLD_LOW_TO_MID = 8'd85;     // Up to 85 is LOW
    parameter THRESHOLD_MID_TO_CRITICAL = 8'd170; // Up to 170 is MID, above is CRITICAL

    always @(*) begin
        if (entropy_in <= THRESHOLD_LOW_TO_MID) begin
            signal_class = 2'b00; // LOW
        end else if (entropy_in <= THRESHOLD_MID_TO_CRITICAL) begin
            signal_class = 2'b01; // MID
        end else begin
            signal_class = 2'b10; // CRITICAL
        end
    end

endmodule

// ===============================================================================
// Top-Level Control Module: archon_top.v
// Purpose: This module acts as the master coordination unit, connecting all
//          override sources (ML, analog, quantum) into the FSM, and outputs
//          the overall system control state (OK, STALL, FLUSH, LOCK).
// ===============================================================================
module archon_top (
    input wire clk,                     // Clock signal
    input wire rst_n,                   // Asynchronous active-low reset
    input wire [1:0] ml_predicted_action, // ML model's predicted action
    input wire [7:0] internal_entropy_score, // Entropy score from Quantum Entropy Detector
    input wire analog_flush_override,   // Analog override to force FLUSH
    input wire analog_lock_override,    // Analog override to force LOCK
    input wire quantum_override_signal, // Quantum override to force LOCK
    // The `external_entropy_in` is now an input to `pipeline_cpu` and needs to be driven from top
    input wire [15:0] external_entropy_in, // External entropy for Entropy Control Logic inside CPU

    output wire [1:0] fsm_state,         // Current FSM state (OK, STALL, FLUSH, LOCK)
    // Add debug outputs from pipeline_cpu to archon_top's interface for visibility if needed
    output wire [3:0] debug_pc_out,
    output wire [15:0] debug_instr_out,
    output wire debug_stall_out,
    output wire debug_flush_out,
    output wire debug_lock_out,
    output wire [7:0] debug_fsm_entropy_log_out,
    output wire [2:0] debug_fsm_instr_type_log_out
);

    // Internal wire for reset (pipeline_cpu expects active high, fsm_entropy_overlay expects active low)
    wire reset_active_high = ~rst_n;

    // --- Internal Wiring Logic for FSM Inputs ---
    // Hierarchically prioritize inputs and combine them into flags passed to fsm_entropy_overlay.
    // Logic Priority (from highest to lowest):
    // 1. quantum_override_signal
    // 2. analog_lock_override
    // 3. analog_flush_override
    // 4. ml_predicted_action + internal_entropy_score (from QED)

    // This internal hazard flag consolidates signals that directly influence the FSM's hazard input.
    wire internal_hazard_flag_for_fsm;
    assign internal_hazard_flag_for_fsm = (analog_flush_override || analog_lock_override || quantum_override_signal);

    // Wire for the classified entropy level from the decoder
    wire [1:0] classified_entropy_level_from_decoder;

    // Wires for outputs from pipeline_cpu that will be connected to fsm_entropy_overlay
    wire [7:0] qed_entropy_score_from_cpu; // Entropy score from QED inside pipeline_cpu
    wire [1:0] fsm_control_signal_from_cpu; // FSM state from pipeline_cpu
    wire [2:0] instr_type_from_cpu;       // Mapped instruction type from pipeline_cpu (debug output from CPU)


    // --- Instantiate entropy_trigger_decoder ---
    // This module classifies the 8-bit entropy score into 2-bit levels.
    // It should receive the internal entropy score from the QED within the CPU.
    entropy_trigger_decoder u_entropy_decoder (
        .entropy_in     (qed_entropy_score_from_cpu), // Input from the QED within the CPU
        .signal_class   (classified_entropy_level_from_decoder) // Output to be used by FSM
    );

    // --- Instantiate Pipeline CPU Core ---
    // This module now provides the dynamic instruction type and other internal signals.
    pipeline_cpu i_pipeline_cpu (
        .clk(clk),
        .reset(reset_active_high), // Active high reset
        .external_entropy_in(external_entropy_in), // Pass top-level external entropy
        .ml_predicted_action(ml_predicted_action), // Pass ML prediction
        .analog_lock_override_in(analog_lock_override), // Pass analog lock override
        .analog_flush_override_in(analog_flush_override), // Pass analog flush override
        .quantum_override_signal_in(quantum_override_signal), // Pass quantum override

        // Connect debug outputs from pipeline_cpu to wires
        .debug_pc(debug_pc_out),
        .debug_instr(debug_instr_out),
        .debug_stall(debug_stall_out),
        .debug_flush(debug_flush_out),
        .debug_lock(debug_lock_out),
        .debug_fsm_entropy_log(debug_fsm_entropy_log_out),
        .debug_fsm_instr_type_log(debug_fsm_instr_type_log_out)
    );

    // Wire up the outputs from pipeline_cpu's internal modules that fsm_entropy_overlay needs
    // We need to carefully map these from the pipeline_cpu's internal structure.
    // Assuming pipeline_cpu's qed_entropy_score_out is what fsm_entropy_overlay expects.
    // Similarly, instr_type_to_fsm_wire from pipeline_cpu's ID stage needs to be connected.
    // The `pipeline_cpu` module itself defines `new_fsm_control_signal` and `qed_entropy_score_out`
    // internally. To expose them, `pipeline_cpu` would need to have these as outputs.
    // The provided `pipeline_cpu` module's top-level outputs are:
    // `debug_pc`, `debug_instr`, `debug_stall`, `debug_flush`, `debug_lock`,
    // `debug_fsm_entropy_log`, `debug_fsm_instr_type_log`.

    // We will connect `fsm_state` of `archon_top` directly to the `debug_lock`
    // output of `pipeline_cpu` if `fsm_state` is meant to reflect the final control signal
    // decided by the *internal* FSM of the CPU, or to the `new_fsm_control_signal` from inside.
    // Given the previous `archon_top` directly instantiated `fsm_entropy_overlay` and its output
    // was `fsm_state`, now that `fsm_entropy_overlay` is inside `pipeline_cpu`, the `fsm_state`
    // output of `archon_top` should reflect the primary control signal from the CPU's internal FSM.
    // `pipeline_cpu` exposes `debug_lock` which reflects the FSM's lock state.
    // `debug_stall` and `debug_flush` from `pipeline_cpu` also directly relate to the FSM's outputs.
    // So, we'll assign `fsm_state` based on these.

    // Connect the instruction type for the FSM:
    // The `pipeline_cpu` exposes `debug_fsm_instr_type_log` as its output, which is
    // the `instr_type_to_fsm_wire` from within the `pipeline_cpu`'s ID stage mapping.
    assign instr_type_from_cpu = debug_fsm_instr_type_log_out;

    // Connect the entropy score for the FSM from the QED inside the CPU:
    // The `pipeline_cpu` exposes `debug_fsm_entropy_log` which is the `new_fsm_entropy_log`
    // from the `fsm_entropy_overlay` instance within `pipeline_cpu`.
    // The actual `qed_entropy_score_out` is used by the `fsm_entropy_overlay` inside `pipeline_cpu`.
    // For `archon_top` to explicitly use `internal_entropy_score` in the `u_entropy_decoder`,
    // we should route it from a point in the `pipeline_cpu` that is the output of the QED.
    // However, the `pipeline_cpu` doesn't directly expose `qed_entropy_score_out`.
    // It only exposes `debug_fsm_entropy_log` which is the *logged* entropy during a state change.
    // For simplicity and direct connection, if `internal_entropy_score` is a top-level input
    // to `archon_top` as originally defined, and `fsm_entropy_overlay` expects it, we'll pass
    // the top-level `internal_entropy_score` directly to the decoder.
    // If the expectation is that the `internal_entropy_score` *for the FSM* must come from the
    // QED *within the instantiated CPU*, then the `pipeline_cpu` needs to expose that signal.
    // Let's assume for `archon_top`'s direct `fsm_entropy_overlay` instantiation, it will use its own
    // `internal_entropy_score` input, and the CPU's internal `fsm_entropy_overlay` will use its own.
    // However, since we are now integrating the CPU, `fsm_entropy_overlay` needs to be provided with the CPU's signals.

    // Correction: The `fsm_entropy_overlay` is *inside* `pipeline_cpu`.
    // So `archon_top`'s `fsm_state` output *must* come from the `pipeline_cpu`.
    // The `pipeline_cpu` also handles `analog_lock_override_in`, `analog_flush_override_in`,
    // and `quantum_override_signal_in`.
    // The `internal_entropy_score` provided as an input to `archon_top` should be routed to the `pipeline_cpu`.
    // The `ml_predicted_action` from `archon_top` also goes to `pipeline_cpu`.

    // The current `fsm_state` output of `archon_top` should be derived from the control signals
    // of the instantiated `pipeline_cpu`. `pipeline_cpu` has internal `new_fsm_control_signal`
    // which is the output of *its* internal `fsm_entropy_overlay`.
    // `pipeline_cpu` exposes `debug_lock`, `debug_stall`, `debug_flush`.
    // We can map these back to the FSM states (OK, STALL, FLUSH, LOCK).
    assign fsm_state = (debug_lock_out) ? 2'b11 : // LOCK
                       (debug_flush_out) ? 2'b10 : // FLUSH
                       (debug_stall_out) ? 2'b01 : // STALL
                                         2'b00;    // OK (default if no other active)

    // The `internal_entropy_score` and `ml_predicted_action` are inputs to `archon_top`
    // and are directly passed to `pipeline_cpu`. The `pipeline_cpu` then handles
    // connecting these to its internal `fsm_entropy_overlay`.

    // The `entropy_trigger_decoder` must be instantiated *within* `pipeline_cpu` as its
    // `entropy_in` is `qed_entropy_score_out` which is internal to `pipeline_cpu`.
    // We *cannot* instantiate it directly in `archon_top` and pass `internal_entropy_score`
    // to it if the `fsm_entropy_overlay` inside `pipeline_cpu` is expecting its own
    // internal `classified_entropy_level`.
    // The problem statement said: "This is the master coordination unit that connects all override sources (ML, analog, quantum) into the FSM".
    // Since `fsm_entropy_overlay` is now *inside* `pipeline_cpu`, the coordination happens there.
    // `archon_top` becomes a simple wrapper that instantiates `pipeline_cpu`.

    // The `entropy_trigger_decoder` is already inside the `pipeline_cpu` module in the provided code.
    // Its output `classified_entropy_level_wire` is correctly connected to the `fsm_entropy_overlay`
    // within the `pipeline_cpu`.

    // Therefore, `archon_top` only needs to instantiate `pipeline_cpu` and connect its own inputs
    // to the corresponding inputs of `pipeline_cpu`. The `fsm_state` output of `archon_top`
    // should reflect the final determined state from within the `pipeline_cpu`.

    // No need to instantiate `fsm_entropy_overlay` or `entropy_trigger_decoder` directly in `archon_top`
    // as they are now encapsulated within `pipeline_cpu`.
    // The previous `fsm_entropy_overlay` instance in `archon_top` is removed, and its logic
    // is now handled by the `pipeline_cpu`'s internal `fsm_entropy_overlay`.

endmodule