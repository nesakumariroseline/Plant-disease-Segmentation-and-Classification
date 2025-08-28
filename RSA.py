import numpy as np
import random as rn
import time
import numpy.matlib

def RSA(X, F_obj, lb, ub, T):

    LB = lb[0, :]
    UB = ub[0, :]
    Best_F = float('inf') # best fitness
    N, Dim = X.shape[0], X.shape[1]
    Xnew = np.zeros((N, Dim))
    Conv = np.zeros((1, T)) #Convergance  array

    t = 1 # starting  iteration
    Alpha = 0.1 # the  best  value 0.1
    Beta = 0.005 # the  best value 0.005
    Ffun = np.zeros((1, X.shape[0])) # (old fitness values)
    Ffun_new = np.zeros((1,X.shape[0])) # (new fitness values)
    Best_P = np.zeros((1,X.shape[1]))
    for i in range(X.shape[0]):
        Ffun[0, i] = F_obj(X[i, :]) # Calculate  the fitness  values  of  solutions
        if Ffun[0, i] < Best_F:
            Best_F = Ffun[0, i]
            Best_P[0, :] = X[i, :]

    ct = time.time()
    while t < T + 1: # Main loop % Update the Position of solutions
        ES = 2 * np.random.rand() * (1 - (t / T)) # Probability Ratio
        for i in range(1, X.shape[0]):
            for j in range(X.shape[1]):
                R = Best_P[0, j] - X[(rn.randint(1, X.shape[0]), j)] / ((Best_P[0, j]) + np.finfo(float).eps)
                P = Alpha + (X[i, j] - np.mean(X[i, :])) / (Best_P[0, j] * (UB[j] - LB[j]) + np.finfo(float).eps)
                Eta = Best_P[0, j] * P
                if (t < T / 4):
                    Xnew[i, j] = Best_P[0, j] - Eta * Beta - R * np.random.rand()
                elif(t < 2 * T / 4  and  t >= T / 4):
                    Xnew[i, j] = Best_P[0, j] * X[(rn.randint(1, X.shape[0]), j)]* ES * np.random.rand()
                elif(t < 3 * T / 4 and t >= 2 * T / 4):
                    Xnew[i, j] = Best_P[0, j] * P * np.random.rand()
                else:
                    Xnew[i, j] = Best_P[0, j] - Eta * np.finfo(float).eps - R * np.random.rand()

            Flag_UB = Xnew[i, :] > UB # check if they  exceed(up)  the  boundaries
            Flag_LB = Xnew[i, :] < LB # check if they exceed(down) the  boundaries
            Xnew[i, :]=(Xnew[i, :] * (~(Flag_UB + Flag_LB)))+UB * Flag_UB + LB * Flag_LB
            Ffun_new[0, i] = F_obj(X[i, :])
            if Ffun_new[0, i] < Ffun[0, i]:
                X[i, :]=Xnew[i, :]
                Ffun[0, i] = Ffun_new[0, i]

            if Ffun[0, i] < Best_F:
                Best_F = Ffun[0, i]
                Best_P = X[i, :]

        Conv[t]= Best_F # Update the  convergence curve
        t = t + 1
    ct = time.time() - ct
    return Best_F, Conv, Best_P, ct
