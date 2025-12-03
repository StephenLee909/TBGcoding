import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple

# 設定輸出資料夾
def setup_output_folder(folder_name: str, subfolders=None) -> str:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name,exist_ok=True)
    
    if subfolders:
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_name, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path,exist_ok=True)

    return folder_name
# 設定參數
def constants():
    a = 2.464  # lattice constant (Å)
    v = np.sqrt(3) / 2 * a * 2.649  # Dirac velocity (J*Å)
    k_D = 4 * np.pi / (3 * a)  # Magnitude of BZ corner wave vector
    w = 110 * 10**-3  # hopping energy (eV)
    return v, k_D, w

# 定義Pauli矩陣
def pauli_matrices():
    s_0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.cdouble)
    s_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
    s_2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
    return s_0, s_1, s_2

def alpha_parameters(alpha_range: torch.Tensor, v: float, k_D: float, w: float, num_path: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_theta_range = w / (v * alpha_range)
    theta_range = 2 * torch.arcsin(k_theta_range / (2 * k_D))
    q_ = torch.tensor([[0, 1], [-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) * k_theta_range.unsqueeze(1).unsqueeze(2)
    q_nn = torch.cat([q_[:, 0:3, :], -q_[:, 0:3, :]], dim=1)

    L = 2.464 / (2 * torch.sin(theta_range / 2) * 3**0.5)
    path = torch.linspace(-1, 2, num_path) * L[:, torch.newaxis]
    
    return theta_range, k_theta_range, q_nn, path

# Convergence length calculation
def calculate_convergence_length(theta_range: torch.Tensor) -> int:
    est_Qnum = 5/np.degrees(torch.min(theta_range).item())**2
    est_length = int(np.rint((est_Qnum*3**1.5/4)**0.5))
    return est_length

# Q point
def generate_Qpts(vector: torch.Tensor,b0: torch.Tensor,b1:torch.Tensor,length: int) -> torch.Tensor:
    vectors_set = set()
    m = int(torch.ceil(2*length/3))+2
    n = int(torch.ceil((2*length+1)/3))+2
    for i in range(-m,n):
        for j in range(-m,n):
            if torch.linalg.norm(vector+i*b0+j*b1) <= length:
                vectors_set.add(tuple(vector+i*b0+j*b1))
    return torch.tensor(list(vectors_set))

# Calculate Charge  
def chrage_indicies(num_Qpts: int, num_QB: int) -> list:
    BA = torch.arange(0, 2*num_QB, 2)
    BB = torch.arange(1, 2*num_QB, 2)
    TA = torch.arange(2*num_QB, 2*num_Qpts, 2)
    TB = torch.arange(2*num_QB+1, 2*num_Qpts, 2)
    return [BA, BB, TA, TB]

def Q_parameters(length: int, k_theta_range: torch.Tensor) -> Tuple[torch.Tensor, list[int], list]:
    q0_ = torch.tensor([[0,1],[-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) #q0_ is unit vector
    b0_ = torch.tensor([q0_[0]-q0_[1],q0_[0]-q0_[2]])
    Q_B = generate_Qpts(-q0_[1], b0_[0], b0_[1], length)
    Q_T = generate_Qpts(torch.zeros(2), b0_[0], b0_[1], length)
    Q0 = torch.cat((Q_B, Q_T), dim=0)
    band_indices = [Q0.shape[0]-1,Q0.shape[0]] #中間2條離0最近
    charge_indicies = chrage_indicies(Q0.shape[0], Q_B.shape[0])
    Q = Q0 * k_theta_range[:,np.newaxis,np.newaxis]
    return Q, band_indices, charge_indicies


# Hamiltonian計算
def h(k, Q,s_1,s_2):
    return (k[0] - Q[0]) * s_1 + (k[1] - Q[1]) * s_2

def T(i, j, Q, q_nn, s_0, s_1, s_2):
    difference = Q[i] - Q[j]
    norm_diffs = [np.linalg.norm(difference - q_nn[k]) for k in range(6)]
    min_index = np.argmin(norm_diffs)

    if norm_diffs[min_index] < 10**-6:
        if min_index in (0,3):
            return s_0 + s_1
        elif min_index in (1,4):
            return s_0 - 0.5 * s_1 + (3**0.5 / 2) * s_2
        else:
            return s_0 - 0.5 * s_1 - (3**0.5 / 2) * s_2
    return np.zeros((2, 2))

def hamiltonian(k, Q, q_nn, v, w, s_0, s_1, s_2):
    H = np.zeros((Q.shape[0]*2, Q.shape[0]*2), dtype=complex)
    # 填入上三角部分
    for i in range(Q.shape[0]):
        for j in range(i + 1, Q.shape[0]):
            H[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = w * T(i, j, Q, q_nn, s_0, s_1, s_2)
    
    # 填充下三角部分
    H += np.conjugate(H).T
    
    # 填對角線
    for i in range(Q.shape[0]):
        H[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = v * h(k, Q[i], s_1, s_2)
    
    return H

# 計算能帶結構
def calculate_band_structure(theta_range, k_points, Q, q_nn, v, w, s_0, s_1, s_2):
    num_t = theta_range.shape[0]
    H_dim = Q.shape[1]*2
    energies = np.zeros((num_t, k_points.shape[1],H_dim)) #(theta,k-points,what band)
    eigenvectors = np.zeros((num_t, k_points.shape[1],H_dim,H_dim), dtype = complex) #(theta,k-points,component of vector,vector at what band)
    H = np.zeros((num_t, k_points.shape[1],H_dim,H_dim), dtype = complex)
    
    for t in range(num_t):
        for k in range(k_points.shape[1]):
            H[t,k] = hamiltonian(k_points[t,k], Q[t], q_nn[t], v, w, s_0, s_1, s_2)
            eigenvalues, eigenvector = np.linalg.eigh(H[t,k])
            energies[t,k,:] = eigenvalues
            eigenvectors[t,k,:,:] = eigenvector
    
    return energies, eigenvectors



def chrage_func(indicies, eigenvector, Q, r):
    z = 0
    for i in indicies:
        z += eigenvector[i] * np.exp(1j*(Q[int(np.floor(i/2)),0]*r))
        # z += eigenvector[i]*np.conjugate(eigenvector[i]) test for vector normalization
    return np.abs(z)**2

def generate_phase_path(SLl,num_points)->int:
    L = SLl/3**0.5
    y = np.linspace(-L, 2*L, num_points) 
    return y #Unit Anstrong

def main():
    # Set GPU as defult device
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 常數
    v, k_D, w = constants()
    s_0, s_1, s_2 = pauli_matrices()

    # Alpha Input
    num_pts = 50
    num_path = 50
    alpha_range = torch.linspace(0.5,3.5,num_pts)  # alpha等差 
    theta_range, k_theta_range, q_nn, path = alpha_parameters(alpha_range, v, k_D, w, num_path)

    # Length Modify
    length = calculate_convergence_length(theta_range[-1:]) # in the unit of 1/Anstrong
    output_folder = f'radius={length}'
    subfolders = ['charge_phase']
    setup_output_folder(output_folder, subfolders)
    print("The Output folder is "f'{output_folder}')

    # Define Qpts
    Q, band_indices, charge_indicies = Q_parameters(length, k_theta_range)

    # Calculate Hamiltonian
    k_points = torch.tensor([[0,0],[-1,0],[3**0.5/2,-0.5]]) * k_theta_range[:, np.newaxis, np.newaxis] # K point used to calculate chrage_func

    # Construct H
    _ , eigenvectors = calculate_band_structure(theta_range, k_points, Q, q_nn, v, w, 0*s_0, s_1, s_2)
    
    #Plot Phase
    X, Y = np.meshgrid(alpha_range, np.linspace(-1,2,num_path))
    # 初始化 Z 矩陣，將每個結果存入這裡以避免重複計算
    Z = np.zeros((X.shape[0], X.shape[1], 4, 4), dtype=float) #(phase, alpha, 原子種類, Kpoint種類)
    
    # 計算每個網格點的Charge值並緩存結果
    for j in range(num_pts):  # alpha
        for k in range(num_path): # phase
            for i in range(4):  # 4個原子種類
                for m in range(3):  # 3個k-point
                    Z[k, j, i, m] = np.sum([
                         chrage_func(charge_indicies[i], eigenvectors[j,m, :, l], Q[j], path[j,k]) for l in band_indices
                    ])
                Z[k, j, i, 3] = Z[k, j, i, 0] + Z[k, j, i, 1]  # K+K'的結果

                # for m in[2]:  # 3個k-point，Modify Here if you want to plot some specific Kpoints only
                #     Z[j, k, i, m] = np.sum([
                #         chrage_func(charge_indicies[i], eigenvectors[m, :, l], Q, r) for l in band_indices
                #     ])

    # 利用已計算的Z矩陣來生成圖

    plot_indicies = [(0,),(2,),(0,1),(2,3),(0,2),(0,1,2,3)]
    plot_indicies_name = ["BA","TA","BOT","TOP","A","TOT"]
    plot_indicies_dict = dict(zip(plot_indicies, plot_indicies_name))

    plot_k_points = [(0,),(1,),(2,),(0,1)]
    plot_k_points_name = ["K","K\'","G","K+K\'"]
    plot_k_points_dict = dict(zip(plot_k_points, plot_k_points_name))

    for i in range(6):  # 6種原子排列組合
        plot_type = plot_indicies_dict[tuple(plot_indicies[i])]
        for m in [3]:  # 4種k-point組合
            K_point_type = plot_k_points_dict[tuple(plot_k_points[m])]
        # for m in [2]:  # 4種k-point組合，Modify Here if you want to plot some specific Kpoints only
            Z_tot = np.zeros((X.shape[0],X.shape[1]), dtype=float)
            for index in plot_indicies[i]:  # 取出所需的原子組合
                Z_tot += Z[:, :, index, m]
            
            # 繪製等高線圖
            plt.contourf(X, Y, np.log10(Z_tot), levels=80, cmap='viridis')
            plt.colorbar(label='log(Charge Density)')
            plt.xlabel('alpha')
            phase_labels = ['AB', 'AA', 'BA', 'AB']
            phase_indices = [-1, 0, 1, 2]
            plt.yticks(phase_indices, phase_labels)
            plt.title(f'Charge Phase {plot_type} at {K_point_type}')
            outputname =f'{output_folder}/charge_phase/BM_{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}_{plot_type}_{K_point_type}_np145.png'
            plt.savefig(outputname, dpi=800)
            plt.close()

            # #單一角度
            # plt.plot(Y[:,0], np.log10(Z_tot[:,0]))
            # plt.xlabel('alpha')
            # phase_labels = ['AB', 'AA', 'BA', 'AB']
            # phase_indices = [-1, 0, 1, 2]
            # plt.xticks(phase_indices, phase_labels)
            # plt.ylabel('log(Charge)')
            # plt.title(f'Charge on Path AB-AA-BA-AB {plot_type} at {K_point_type}')
            # plt.grid(True)
            # plt.savefig(f'{output_folder}/charge_phase/path_vs_charge_{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}_{plot_type}_{K_point_type}.png')
            # plt.close()



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} s")

        
# 建立 band class
# 計算函數: 有 calculate_band_structure, calculate_band_slope
# 畫圖函數: plot_band_structure, plot_velocity
# 變數: length, 常數, theta部分, Q部分