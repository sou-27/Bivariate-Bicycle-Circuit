import numpy as np
import stim
import galois
from typing import Literal
from itertools import chain
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

type check_matrix = tuple[tuple[Literal[0,1], int]]

GF2 = galois.GF(2)



def CNOT_pairs(controls, targets):
    """
    Pairs up controls and targets into single interleaved list.
    Eg. controls = [0,1] targets = ['a', 'b'] ---> returns [0,'a',1,'b']
    Used to create control-target structure for CNOT gates in stim.circuit

    Parameters:
    controls (list) : List of controls for CNOT gate
    targets(list) :  List of targets for CNOT gate

    Returns: [List] List consisting of controls and targets in interleaved form.
    """
    return list(chain.from_iterable(zip(controls,targets)))

def create_CNOT_string(register, controls, targets):
     """
     Takes in a register ("L" or "R") and controls and targets and returns approprate control-target
     list for use in CNOT gate. If controls(or targets) is given as a matrix, determines appropriate position of 
     data qubits by looking for 1s in the matrix. The register determines specific index of data qubit
     in circuit.

     Parameters:
     register(str) : "L" or "R"
     controls(List or np.ndarray) : Controls for CNOT gate. If np.ndarray, qubit indices are calculated.
     targets(List or np.ndarray) : Targets for CNOT gate. If np.ndarray, qubit indices are calculated.

     Returns: [List] List consisting of controls and targets in interleaved form.
     """
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
    """
    Class defining a Bivariate Bicycle code given input parameters.
    
    Attributes:
    -----------
    l(int) : Dimension of matrices. 
    m(int) : Dimension of matrices
    A_vars(List[List[int,int]]) : Input trinomial, in format [0 for x;1 for y, power]. Eg. A_vars = [[0,1],[1,1],1,2] => A = x + y + y^2
    B_Vars(List[List[int,int]]) : See above
    p(float) : Fault probability in circuit. Assumed depolarizing noise.
    rounds(int) : Number of rounds of syndrome measurement.
    code_capacity(bool) : True for code capacity noise. Else we use circuit level noise.
    A(np.ndarray) = A_1 + A_2 + A_3
    B(np.ndarray) = B_1 + B_2 + B_3
    HX(np.ndarray) : X-stabilizers. Defined HX = [A|B]
    HZ(np.ndarray) : Z-stabilizers. Defined HZ = [B.T|A.T]
    n(int) : Number of data qubits.
    k(int) : Number of encoded logical qubits.
    kernels(List[np.ndarray]) : Each array is a vector in the space ker(A) intercept ker(B). Usefeul for calculating logical observables.
    circuit(stim.circuit) : BB circuit generated from input parameters.

    calculate_distance() : Calculates code distance for input distance under code capacity conditions. Uses a SAT solver. Fairly 
                            costly computation so only computed on request.

    """
    def __init__(self,*, l :int, m :int, A_vars :check_matrix, B_vars :check_matrix, p:float, rounds :int, code_capacity = False):
        self.l = l
        self.m = m
        self.rounds = rounds
        self.p = p

        if not (len(A_vars) == 3 and len(B_vars) == 3):
            raise ValueError("Invalid input check matrices")
         

        p_l = [(i+1)%l for i in range(l)]
        p_m = [(i+1)%m for i in range(m)]

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

        self.circuit = self.generate_noisy_circuit(code_capacity)

    def calculate_distance(self):
        """
        Calculates code distance of BB circuit under code capacity conditions using SAT optimization.

        Returns : distance(int) - Calculated code distance.
        """
        code_capacity_circuit = self.generate_noisy_circuit(True)
        wdimacs_string = code_capacity_circuit.shortest_error_sat_problem(format="WDIMACS")

        wcnf = WCNF(from_string=wdimacs_string)

        with RC2(wcnf) as rc2:
            solution = rc2.compute()  # Finds the optimal assignment of variables
            distance = rc2.cost 
        return distance

    def get_logical_observables(self):
        """
        Computes pauli operators corresponding to logical Z-observables.

        Returns: List[List[int]] - List containing observables. Each observable is a list of indices where a pauli-Z must be applied.
        """
        kernels = self.kernels
        n = self.n
        shift = int(n/2)

        observables1 = [np.where(kernel == 1)[0].tolist() for kernel in kernels]
        observables2 = [(shift+np.where(kernel == 1)[0]).tolist() for kernel in kernels]

        return observables1 + observables2
         

    def generate_noisy_circuit(self, code_capacity = False):
        """
        Generates BB circuit under given code parameters.

        Parameters:
        code_capacity(bool) : If True, use code capacity conditions. Else use circuit level noise.

        Returns:
        circuit(stim.circuit) : BB circuit.
        """
        n = self.n
        k = self.k
        rounds = self.rounds
        p_measure = self.p
        p_idle = self.p

        if code_capacity:
            p_measure = 0

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
        def _create_round_circuit(p_m, p_i, detectors = False):
            round_circuit = stim.Circuit()
            
            #Step 1-------

            round_circuit.append("R",X_ancillas)
            round_circuit.append("X_ERROR",X_ancillas,p_m)
            round_circuit.append("CX", R_A1_Z)
            round_circuit.append("DEPOLARIZE2", R_A1_Z, p_m)

            #Step 2-------

            round_circuit.append("CX",L_A2_X)
            round_circuit.append("DEPOLARIZE2", L_A2_X, p_m)
            round_circuit.append("CX", R_A3_Z)
            round_circuit.append("DEPOLARIZE2", R_A3_Z, p_m)

            #Step 3--------

            round_circuit.append("CX",R_B2_X)
            round_circuit.append("DEPOLARIZE2", R_B2_X, p_m)
            round_circuit.append("CX", L_B1_Z)
            round_circuit.append("DEPOLARIZE2", L_B1_Z, p_m)

            #Step 4--------

            round_circuit.append("CX",R_B1_X)
            round_circuit.append("DEPOLARIZE2", R_B1_X, p_m)
            round_circuit.append("CX", L_B2_Z)
            round_circuit.append("DEPOLARIZE2", L_B2_Z, p_m)

            #Step 5--------

            round_circuit.append("CX",R_B3_X)
            round_circuit.append("DEPOLARIZE2", R_B3_X, p_m)
            round_circuit.append("CX", L_B3_Z)
            round_circuit.append("DEPOLARIZE2", L_B3_Z, p_m)

            #Step 6--------

            round_circuit.append("CX",L_A1_X)
            round_circuit.append("DEPOLARIZE2", L_A1_X, p_m)
            round_circuit.append("CX", R_A2_Z)
            round_circuit.append("DEPOLARIZE2", R_A2_Z, p_m)

            #Step 7--------

            round_circuit.append("CX",L_A3_X)
            round_circuit.append("DEPOLARIZE2", L_A3_X, p_m)
            round_circuit.append("M", Z_ancillas, p_m)
            round_circuit.append("DEPOLARIZE1", R_qubits, p_i)

            #Step 8--------

            round_circuit.append("M", X_ancillas, p_m)
            round_circuit.append("R",Z_ancillas)
            round_circuit.append("X_ERROR",Z_ancillas,p_m)
            round_circuit.append("DEPOLARIZE1", L_qubits, p_i)
            round_circuit.append("DEPOLARIZE1", R_qubits, p_i)

            if detectors:
                for i in range(n):
                     #First n/2 detectors are for Z-checks and next n/2 detectors are for X-checks
                     round_circuit.append("DETECTOR", [stim.target_rec(-2*n + i), stim.target_rec(-n+i)])

            round_circuit.append("TICK")

            return round_circuit

        #-----------Initializing circuit with perfect measurements. No detectors at this stage------------------

        circuit = stim.Circuit()
        circuit += _create_round_circuit(0,0,False)

        #--------Now implementing rounds of faulty syndrome measurement-----------------------------------------

        circuit += _create_round_circuit(p_measure,p_idle,True) * rounds

        #--------Final measurement readout----------------------------------------------------------------------

        circuit.append("M", data_qubits)

        #--------Final Z-check detectors must be calculated directly from data qubits---------------------------

        Z_checks = {
            Z_ancillas[i] : np.where(HZ[i,:] == 1)[0].tolist() for i in range(int(n/2))
        }

        for i,ancilla in enumerate(Z_ancillas):
            support = Z_checks[ancilla]
            targets = [stim.target_rec(-n + supp) for supp in support]
            targets.append(stim.target_rec(-2*n + i))

            circuit.append("DETECTOR", targets)

        #------Defining logical observables -------------------------------------------------------------------

        for i, observable in enumerate(observables):
            targets = [stim.target_rec(-n+supp) for supp in observable] 

            circuit.append("OBSERVABLE_INCLUDE",targets,i)


        return circuit           






        


        


        






