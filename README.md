# schur-transform
INCOMPLETE REPOSITORY

Computes an efficient quantum algorithm for the quantum Schur transform on $n$ qubits.

The quantum algorithm is returned as a list of $O(n^3)$ two-level operations in matrix form. These could each be decomposed into $O(n\log(n/\epsilon))$ Clifford+T operators using a universal unitary decomposition algorithm such as given in arXiv:1306.3200, for a total operation count of $O(n^4\log(n/\epsilon))$ and overall error bounded by $\epsilon$.

Runs with Mathematica 10.4+.

The quantum algorithm uses $2\lfloor\log_2(n)\rfloor-1$ ancillary qubits for a total register of $n+2\lfloor\log_2(n)\rfloor-1$ qubits. The Schur transform maps the standard computational basis to a basis composed of a direct sum of irreducible modules for the unitary and symmetric groups. Equivalently, it maps individual spin eigenvectors to eigenvectors of the composite spin for the whole register.

We assume that the initial register has the form $(\text{ancillary qubits})\otimes(\text{computational qubits})$. The output register will be organized into subspaces with distinct values of composite total spin. We provide an auxiliary algorithm to calculate the locations of the computational output states.

The main procedure in "Schur Transform.nb" is SchurDecomp[n],  which returns a list of $O(n^3)$ elements with form $entry_i={t_i,R_i,b_i}$. Each $R_i$ is a two-level rotation such that $\prod_i\textbf{I}_{2^{t_i}}\otimes R_i\otimes\textbf{I}_{2^{b_i}}$.

To calculate the rows used to encode the computational output states, run SchurIndices[n]. The output is a list of the rows used to encode the computational output states.
