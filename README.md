# schur-transform
INCOMPLETE REPOSITORY

Computes an efficient quantum algorithm for the quantum Schur transform on $n$ qubits.

The quantum algorithm is returned in the form of a list of $O(n^3)$ two-level operations in matrix form. These could each be decomposed into $O(n\log(n/\epsilon))$ Clifford+T operators using a universal unitary decomposition algorithm such as given in arXiv:1306.3200, for a total operation count of $O(n^4\log(n/\epsilon))$ and overall error bounded by $\epsilon$.

Runs with Mathematica 10.4+.

The quantum algorithm uses $2\lfloor\log_2(n)\rfloor-1$ ancillary qubits for a total register of $n+2\lfloor\log_2(n)\rfloor-1$ qubits. Assumes that the initial register is in the form $(\text{ancillary qubits})\otimes(\text{computational qubits})$.
