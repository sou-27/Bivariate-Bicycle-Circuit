import numpy as np
import stim
import galois
from typing import Literal

type check_matrix = tuple[tuple[Literal[0,1], int]]

GF2 = galois.GF(2)

def term(basis, vec):
            """
            Returns a string representing pauli operator on given support. Eg. for basis = "X"
            and vec = [0,1,1,0,0], returns "X1*X2"

            Parameters:
            basis (str) : Pauli operator to be used
            vec (List of integers) : Support of pauli operator

            Returns:
            String representing given pauli operator.
            """
            return "*".join(f"{basis}{q}" for q in range(len(vec)) if vec[q]==1)

class BivBic:
    def __init__(self, l :int, m :int, A_vars :check_matrix, B_vars :check_matrix):
        self.l = l
        self.m = m

        if not (len(A_vars) == 3 and len(B_vars) == 3):
            raise ValueError("Invalid input check matrices")
         

        p_l = [(i+1)%l for i in range(l)]
        p_m = [(i+1)%l for i in range(m)]

        I_l = GF2.Identity(l)
        I_m = GF2.Identity(m)
        
        S_l = GF2.Identity(l)[p_l]
        S_m = GF2.Identity(m)[p_m]

        x = np.kron(S_l,I_m)
        y = np.kron(I_l,S_m)

        perms = [x,y]

        A = GF2.Zeros(x.shape)
        B = GF2.Zeros(x.shape)

        for mat, p in A_vars:
            A += np.linalg.matrix_power(perms[mat],p)

        for mat, p in B_vars:
            B += np.linalg.matrix_power(perms[mat], p)

        self.A = A
        self.B = B

        HX = np.hstack([A, B])
        HZ = np.hstack([B.T, A.T])

        self.HX = HX
        self.HZ = HZ

        self.n = 2*l*m

        kernels = np.vstack([A,B]).null_space()

        self.kernels = kernels

        self.k = 2 * len(kernels)

        self.circuit = self.generate_noiseless_circuit()

    def get_logical_observables(self):
        kernels = self.kernels
        n = self.n
        triv = GF2.Zeros((1,int(n/2)))
        observables_1 = [term("Z",np.hstack([vec,triv[0]])) for vec in kernels]
        observables_2 = [term("Z",np.hstack([triv[0],vec])) for vec in kernels]

        return " ".join(ob for ob in observables_1+observables_2)
         
    def generate_noiseless_circuit(self):
        HX = self.HX
        HZ = self.HZ
        n = self.n
        k = self.k

        L = []
        Z_checks = " ".join(term("Z",vec) for vec in HZ)
        X_checks = " ".join(term("X",vec) for vec in HX)

        nchecks = n+k


        observables = self.get_logical_observables()

        L.append("MPP "+Z_checks+" "+X_checks+" "+observables)
        L.append("TICK")
        L.append("MPP "+Z_checks+" "+X_checks+" "+observables)
        for i in range(n):
             L.append(f"DETECTOR rec[{-2*nchecks+i}] rec[{-nchecks+i}]")

        for j in range(k):
             L.append(f"OBSERVABLE_INCLUDE({j}) rec[{-2*nchecks + n + j}] rec[{-nchecks+ n + j}]")

        return stim.Circuit("\n".join(L))

