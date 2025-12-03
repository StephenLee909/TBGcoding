import numpy as np
import matplotlib.pyplot as plt

t_pi = 1
a = 1
a_ = a * np.array([[3**0.5 / 2, -0.5], [3**0.5 / 2, 0.5]])
b_ = 4 * np.pi / (3**0.5 * a) * np.array([[0.5, 3**0.5 / 2], [0.5, -3**0.5 / 2]])
e_p = 0 
d_ = a / 3**0.5 * np.array([[1,0],[-0.5, 3**0.5 / 2],[-0.5, -3**0.5 / 2]])

def R(t):
    R = np.array([[np.cos(t),-np.sin(t)],
                  [np.sin(t),np.cos(t)]])
    return R
## k-points

num_points = 50
k_points = []
# Gamma -> M
for i in range(num_points):
    point = (i / num_points) * 2 * np.pi / (3**0.5 * a) * np.array([1, 0])
    k_points.append(point)
# M->K
for i in range(num_points):
    point = (1 - i / num_points) * 2 * np.pi / (3**0.5 * a) * np.array([1, 0]) +i / num_points* 4 * np.pi / (3 * a) * np.array([3**0.5 / 2, 0.5])
    k_points.append(point)
# K-> Gamma
for i in range(num_points):
    point = (1 - i / num_points) * 4 * np.pi / (3 * a) * np.array([3**0.5 / 2, 0.5])
    k_points.append(point)

k_points = np.array(k_points)


# Around BZ
# K = 4 * np.pi / (3 * a) * np.array([0, 1])
# for j in range(6):
#     for i in range(num_points):
#         point = (1 - i / num_points) * K +i / num_points * R(np.pi/3) @ K
#         k_points.append(point)
#     K = R(np.pi/3) @ K

k_points = np.array(k_points)

## Construct the Hamiltonian

def f_k(k):
    # f_k = 1 + np.exp(1j*np.dot(k,a_[0])) +np.exp(1j*np.dot(k,a_[1]))
    f_k = 0
    for i in range(3):
        f_k += np.exp(1j*np.dot(k,d_[i]))

    return f_k

def hamiltonian(k):
    # 將子矩陣填入對角線位置
    H = np.zeros((2, 2), dtype=complex)
    H[0,0] = e_p
    H[1,1] = e_p
    H[0,1] = t_pi * f_k(k)
    H[1,0] = t_pi * np.conjugate(f_k(k))
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
k_labels = ['$\Gamma$', 'M', 'K', '$\Gamma$']
k_indices = [0, num_points, 2 * num_points, 3 * num_points -1]
for index in k_indices:
    plt.axvline(x=index, color='k', linestyle='--', linewidth=0.5)

# 添加標籤
plt.xticks(k_indices, k_labels)

plt.xlabel('k')
plt.ylabel('Energy')
plt.title('Band Structure along $\Gamma$-M-K-$\Gamma$')
plt.legend()
plt.show()

