import numpy as np
import matplotlib.pyplot as plt

## Initial Parameters

# lattice constant (Anstrong)
a = 2.464
# Dirac velocity (J*Anstrong)
v = np.sqrt(3)/2*a*2.649
# rotation degree (Rad)
theta = 5 * np.pi / 180
# Magnitude of BZ corner wave vector 
k_D = 4 * np.pi / (3 * a)
# Morie Lattice constant
k_theta = k_D * np.sin(theta / 2)
# hopping energy (meV)
w = 110 * 10**-3
# Real Space lattice vector a_i And Reciprocal Space lattice vector b_j
a_ = a * np.array([[-0.5, 3**0.5 / 2], [0.5, 3**0.5 / 2]])
b_ = 4 * np.pi / (3**0.5 * a) * np.array([[3**0.5 / 2, 0.5], [-3**0.5 / 2, 0.5]])
# Moiré Lattice vector
q_ = k_theta * np.array([[0, -1], [3**0.5 / 2, 0.5], [-3**0.5 / 2, 0.5]])
# Pauli Matrices
s_0 = np.array([[1,0],[1,0]])
s_1 = np.array([[0,1],[1,0]])
s_2 = np.array([[0,-1j],[1j,0]])
s_3 = np.array([[1,0],[0,-1]])


## k-points

num_points = 50
k_points = []
# A->B
for i in range(num_points):
    point = (1 - i / num_points) * q_[0]
    k_points.append(point)
# B->C
for i in range(num_points):
    point = -i / num_points * q_[0]
    k_points.append(point)
# C->D
for i in range(num_points):
    point = (1 - i / num_points) * -q_[0] + (i / num_points) * (q_[0] + q_[1])
    k_points.append(point)
# D->A
for i in range(num_points):
    point = (1 - i / num_points) * (q_[0] + q_[1]) + (i / num_points) * q_[0]
    k_points.append(point)

k_points = np.array(k_points)

## Submatrices of Hamiltonian

# Intralayer term
def h(k, theta):
    if not np.array_equal(k, np.array([0, 0])):
        theta_k = np.arccos(k[0] / np.linalg.norm(k))
        h = -v * np.linalg.norm(k) * np.array([[0, np.exp(1j * (theta_k - theta))],
                                               [np.exp(-1j * (theta_k - theta)), 0]])
        return h
    else:
        return np.zeros((2, 2))

# Interlayer term
def T(i):
    if i == 0:
        T = s_0 + s_1
        return T
    elif i == 1:
        T = s_0 - 0.5 * s_1 + 3**0.5/2 * s_2
        return T
    elif i == 2:
        T = s_0 - 0.5 * s_1 - 3**0.5/2 * s_2
        return T
    else:
        return np.zeros((2, 2))


## Construct the Hamiltonian

def hamiltonian(k):
    # 將子矩陣填入對角線位置
    H = np.zeros((8, 8), dtype=complex)
    
    H[0:2, 0:2] = h(k, theta / 2)
    for i in range(1, 4):
        H[2 * i:2 * i + 2, 2 * i:2 * i + 2] = h(k - q_[i - 1], -theta / 2)

    # # 填入第一列（保持不變）
    # for i in range(1, 4):
    #     H[0:2, i * 2:i * 2 + 2] = w*T(i - 1)

    # # 填入第一行（進行共軛轉置）
    # for i in range(1, 4):
    #     H[i * 2:i * 2 + 2, 0:2] = w*np.conjugate(T(i - 1)).T

    return H


## 計算能帶結構

def calculate_band_structure(k_points):
    energies = []
    for k in k_points:
        H = hamiltonian(k)
        eigenvalues, _ = np.linalg.eigh(H)
        energies.append(eigenvalues)
    return np.array(energies)

energies = calculate_band_structure(k_points)


## 繪製能帶結構

plt.figure(figsize=(10, 6))
for i in range(energies.shape[1]):
    plt.plot(energies[:, i], label=f'Band {i + 1}')

# 添加標記點
k_labels = ['A', 'B', 'C', 'D', 'A']
k_indices = [0, num_points, 2 * num_points, 3 * num_points, 4 * num_points - 1]
for index in k_indices:
    plt.axvline(x=index, color='k', linestyle='--', linewidth=0.5)

# 添加標籤
plt.xticks(k_indices, k_labels)

plt.xlabel('k')
plt.ylabel('Energy')
plt.title('Band Structure along A-B-C-D-A')
plt.legend()
plt.show()