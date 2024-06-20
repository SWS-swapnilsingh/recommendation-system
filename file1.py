import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random


df = pd.read_csv('recommendation-system.csv', index_col=0)
# print(df.head(5))

G = nx.DiGraph()

# Add nodes
for node in df.index:
    G.add_node(node)

#droping first row
# df = df.drop(index=[0], axis = 0)
# df = df.drop(df.columns[0], axis = 1)

# print(df.head())
# df.head(5)


# Add edges based on the DataFrame values
for row in df.index:
    for col in df.columns:
        if pd.notna(df.at[row, col]):  # Check if the cell is not NaN (blank)
            G.add_edge(col, row)

# Adjacency matrix
M = np.array([[0 for x in range(len(df.index))] for y in range(len(df.index))])
# print(M)
# print(df.iloc[0, :])
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        if df.iloc[i, j] != ' ':
#             # G.add_edge(df.index[i], df.columns[j])
            # print('not null', str(df.iloc[i, j]),(i,j))
            M[i, j] = 1
        elif df.iloc[i, j] == ' ':
            # print('null hai', str(df.iloc[i, j]), (i,j))
            M[i, j] = 0

#adding -1 value in M matrix where M[i,j] = 0 and M[j,i] = 1
for i in range(len(M)):
    for j in range(len(M[0])):
        if M[i, j] == 1 and M[i, j] == 0:
            M[i, j] = -1
# print(M.shape)

# print(M[128, :])
# print(df.iloc[131, 25] == ' ')

# Value of k
K = 30

# U is 133 X 30 matrix
U = np.array([[random.randint(-1,1) for _ in range(K)] for _ in range(133)])
# print(U)

# V is 30 X 30 matrix
V = np.array([[random.randint(-1, 1) for _ in range(133)] for _ in range(K)])
# print(M.shape)

def error_matrix_and_total_error(M, generated_matrix):
    # Calculate error matrix
    E = M - generated_matrix
    # Calculate error matrix squared
    E_squared = E**2
    # Calculate total error
    total_error = 0
    for i in range(len(E)):
        for j in range(len(E[0])):
            total_error += E_squared[i, j]
    return E, total_error

# function to update U and V which accepts i,j th error value
def update_U_V(alpha, i, j, error, U, V):
    # Update values of U
    for k in range(K):
        U[i, k] = U[i, k] - alpha*2*error*V[k, j]
    # Update values of V
    for k in range(K):
        V[k, j] = V[k, j] - alpha*2*error*U[i, k]
    return U, V


alpha = 0.001
l = 0
total_error_list = []
iteration_list = []
# loop
while l < 10:
    # print(l)
    generated_matrix = U@V
    # print(generated_matrix)
    E, total_error = error_matrix_and_total_error(M, generated_matrix)
    total_error_list.append(total_error)
    # print(total_error)

    # iterate over E matrix
    for i in range(len(E)):
        for j in range(len(E[0])):
            if E[i, j]!= 0:
                error = E[i, j]
                U_, V_ = update_U_V(alpha, i, j, error, U, V)
                U = U_
                V = V_
    iteration_list.append(l)
    l += 1

print(total_error_list)
