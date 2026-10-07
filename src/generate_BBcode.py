import numpy as np
import stim
import galois
from typing import Literal
from itertools import chain

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

def CNOT_pairs(controls, targets):
    return list(chain.from_iterable(zip(controls,targets)))

def create_CNOT_string(register, controls, targets):
     N = len(controls) 
     shift = 0
     if register == "R":
          shift = N
        
     if type(controls) == list:
        #If controls are in a list => CNOT ancilla data_qubit
        trgts = []
        for i in range(N):
            trgts.append(shift + int(np.argmax(targets[i,:])))
             

        return CNOT_pairs(controls, trgts)
     else:
        #If controls are not in a list => targets are in a list => CNOT data_qubit ancilla
        ctrls = []
        for i in range(N):
            ctrls.append(shift + int(np.argmax(controls[i,:])))

        return CNOT_pairs(ctrls, targets)

    
class BivBic:
    def __init__(self, l :int, m :int, A_vars :check_matrix, B_vars :check_matrix, p:float, rounds :int):
        self.l = l
        self.m = m
        self.rounds = rounds
        self.p = p

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

        A1 = np.linalg.matrix_power(perms[A_vars[0][0]],A_vars[0][1])
        A2 = np.linalg.matrix_power(perms[A_vars[1][0]],A_vars[1][1])
        A3 = np.linalg.matrix_power(perms[A_vars[2][0]],A_vars[2][1])

        B1 = np.linalg.matrix_power(perms[B_vars[0][0]],B_vars[0][1])
        B2 = np.linalg.matrix_power(perms[B_vars[1][0]],B_vars[1][1])
        B3 = np.linalg.matrix_power(perms[B_vars[2][0]],B_vars[2][1])

        A = A1 + A2 + A3
        B = B1 + B2 + B3

        self.A = A
        self.B = B
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

        HX = np.hstack([A, B])
        HZ = np.hstack([B.T, A.T])

        self.HX = HX
        self.HZ = HZ

        self.n = 2*l*m

        kernels = np.vstack([A,B]).null_space()

        self.kernels = kernels

        self.k = 2 * len(kernels)

        #self.circuit = self.generate_noiseless_circuit()

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

    def generate_noisy_circuit(self):
        n = self.n
        k = self.k
        rounds = self.rounds
        p = self.p

        HX = self.HX
        HZ = self.HZ
        A1 = self.A1
        A2 = self.A2
        A3 = self.A3
        B1 = self.B1
        B2 = self.B2
        B3 = self.B3

        observables = self.get_logical_observables()


        data_qubits = [i for i in range(n)]
        L_qubits = [i for i in range(int(n/2))]
        R_qubits = [i for i in range(int(n/2),n)]
        X_ancillas = [i for i in range(n,int(3*n/2))]
        Z_ancillas = [i for i in range(int(3*n/2), (2*n))]


        R_A1_Z = create_CNOT_string("R", A1.T, Z_ancillas)
        R_A2_Z = create_CNOT_string("R", A2.T, Z_ancillas)
        R_A3_Z = create_CNOT_string("R", A3.T, Z_ancillas)
        L_B1_Z = create_CNOT_string("L", B1.T, Z_ancillas)
        L_B2_Z = create_CNOT_string("L", B2.T, Z_ancillas)
        L_B3_Z = create_CNOT_string("L", B3.T, Z_ancillas)

        L_A1_X = create_CNOT_string("L", X_ancillas, A1)
        L_A2_X = create_CNOT_string("L", X_ancillas, A2)
        L_A3_X = create_CNOT_string("L", X_ancillas, A3)
        R_B1_X = create_CNOT_string("R", X_ancillas, B1)
        R_B2_X = create_CNOT_string("R", X_ancillas, B2)
        R_B3_X = create_CNOT_string("R", X_ancillas, B3)
        
        #----------Defining circuit for a single round of syndrome measurement----------------------------------
        def _create_round_circuit(prob = p, detectors = False):
            round_circuit = stim.Circuit()
            
            #Step 1-------

            round_circuit.append("R",X_ancillas,p)
            round_circuit.append("CX", R_A1_Z)
            round_circuit.append("DEPOLARIZING2", R_A1_Z, prob)

            #Step 2-------

            round_circuit.append("CX",L_A2_X)
            round_circuit.append("DEPOLARIZING2", L_A2_X, prob)
            round_circuit.append("CX", R_A3_Z)
            round_circuit.append("DEPOLARIZING2", R_A3_Z, prob)

            #Step 3--------

            round_circuit.append("CX",R_B2_X)
            round_circuit.append("DEPOLARIZING2", R_B2_X, prob)
            round_circuit.append("CX", L_B1_Z)
            round_circuit.append("DEPOLARIZING2", L_B1_Z, prob)

            #Step 4--------

            round_circuit.append("CX",R_B1_X)
            round_circuit.append("DEPOLARIZING2", R_B1_X, prob)
            round_circuit.append("CX", L_B2_Z)
            round_circuit.append("DEPOLARIZING2", L_B2_Z, prob)

            #Step 5--------

            round_circuit.append("CX",R_B3_X)
            round_circuit.append("DEPOLARIZING2", R_B3_X, prob)
            round_circuit.append("CX", L_B3_Z)
            round_circuit.append("DEPOLARIZING2", L_B3_Z, prob)

            #Step 6--------

            round_circuit.append("CX",L_A1_X)
            round_circuit.append("DEPOLARIZING2", L_A1_X, prob)
            round_circuit.append("CX", R_A2_Z)
            round_circuit.append("DEPOLARIZING2", R_A2_Z, prob)

            #Step 7--------

            round_circuit.append("CX",L_A3_X)
            round_circuit.append("DEPOLARIZING2", L_A3_X, prob)
            round_circuit.append("M", Z_ancillas, prob)
            round_circuit.append("DEPOLARIZING1", R_qubits, prob)

            #Step 8--------

            round_circuit.append("M", X_ancillas, prob)
            round_circuit.append("R",Z_ancillas,prob)
            round_circuit.append("DEPOLARIZING1", L_qubits, prob)
            round_circuit.append("DEPOLARIZING1", R_qubits, prob)

            if detectors:
                 round_circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
                 round_circuit.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-4)])

            round_circuit.append("TICK")

            return round_circuit

        #-----------Initializing circuit with perfect measurements. No detectors at this stage------------------

        circuit = stim.Circuit()
        circuit += _create_round_circuit(0,False)

        #--------Now implementing rounds of faulty syndrome measurement-----------------------------------------

        circuit += _create_round_circuit(p,True) * rounds

        



        


        


        






