# schur-transform
INCOMPLETE REPOSITORY

Computes an efficient quantum algorithm for the quantum Schur transform on $n$ qubits.

Runs with Mathematica 10.4+.

The quantum algorithm uses $2\lfloor\log_2(n)\rfloor-1$ ancillary qubits for a total register of $n+2\lfloor\log_2(n)\rfloor-1$ qubits. Assumes that the initial register is in the form $(\text{ancillary qubits})\otimes(\text{computational qubits})$.
