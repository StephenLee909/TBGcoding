import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import time
from scipy.signal import argrelextrema

# 設定輸出資料夾
def setup_output_folder(folder_name, subfolders=None):
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
    w = 110 * 10**-3  # hopping energy (meV)
    return v, k_D, w

# 定義Pauli矩陣
def pauli_matrices():
    s_0 = np.array([[1, 0], [0, 1]])
    s_1 = np.array([[0, 1], [1, 0]])
    s_2 = np.array([[0, -1j], [1j, 0]])
    return s_0, s_1, s_2

def theta_parameter(theta, v, k_D, w):
    k_theta = 2 * k_D * np.sin(theta / 2)
    q_ = k_theta * np.array([[0, 1], [-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]])
    alpha = w / (v * k_theta)
    b_ = np.array([q_[0]-q_[2],q_[1]-q_[2]])
    return k_theta, q_, alpha, b_

# K point
def generate_kpts(k_paths, num_kpts):
    k_points = [((1 - i / num_kpts) * start + (i / num_kpts) * end) for start, end in k_paths for i in range(num_kpts)]
    return np.array(k_points)

# Q point
def generate_Qpts(vector: np.ndarray, b0: np.ndarray, b1: np.ndarray,   length: np.ndarray) -> np.ndarray:
    vectors_set = set()
    m = int(np.ceil(2*length/3))+2
    n = int(np.ceil((2*length+1)/3))+2
    for i in range(-m,n):
        for j in range(-m,n):
            if np.linalg.norm(vector+i*b0+j*b1) <= length:
                vectors_set.add(tuple(vector+i*b0+j*b1))
    return np.array(list(vectors_set))

# Convergence length calculation
def calculate_convergence_length(theta_range: np.ndarray) -> np.ndarray:
    est_Qnum = 5/np.degrees(theta_range)**2
    est_length = (est_Qnum*3**1.5/np.pi)**0.5
    return est_length

# Hamiltonian計算
def h(k, Q,s_1,s_2):
    return (k[0] - Q[0]) * s_1 + (k[1] - Q[1]) * s_2

def T(i, j, Q, q_nn, s_0, s_1, s_2):
    difference = Q[i] - Q[j]
    norm_diffs = [np.linalg.norm(difference - q_nn[k]) for k in range(6)]
    min_index = np.argmin(norm_diffs)

    if norm_diffs[min_index] < 10**-6:
        if min_index < 2:
            return s_0 + s_1
        elif min_index < 4:
            return s_0 - 0.5 * s_1 + (3**0.5 / 2) * s_2
        else:
            return s_0 - 0.5 * s_1 - (3**0.5 / 2) * s_2
    return np.zeros((2, 2))

def hamiltonian(k, Q, q_nn, v, w, s_0, s_1, s_2,num_Qpts):
    H = np.zeros((num_Qpts*2, num_Qpts*2), dtype=complex)
    # 填入上三角部分
    for i in range(num_Qpts):
        for j in range(i + 1, num_Qpts):
            H[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = w * T(i, j, Q, q_nn, s_0, s_1, s_2)
    
    # 填充下三角部分
    H += np.conjugate(H).T
    
    # 填對角線
    for i in range(num_Qpts):
        H[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = v * h(k, Q[i], s_1, s_2)
    
    return H

# 計算能帶結構
def calculate_band_structure(k_points, Q, q_nn, v, w, s_0, s_1, s_2, num_Qpts):
    energies = []
    eigenvectors_list = []
    
    for k in k_points:
        H = hamiltonian(k, Q, q_nn, v, w, s_0, s_1, s_2, num_Qpts)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        energies.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)
    
    return np.array(energies), np.array(eigenvectors_list)

# 計算能帶斜率絕對值，再取平均
def calculate_band_slope(energies, k_points, band_indices):
    slopes = []
    dk = np.linalg.norm(k_points[1] - k_points[0]) # 每個k間距離都等長，故取第二個跟第一個的距離
    for i in band_indices:
        band = energies[:, i]
        slope = np.gradient(band, dk)
        slopes.append(np.abs(slope))
    return np.mean(slopes)

# Plot velocity v.s theta(degree) and alpha_square
def plot_velocity(theta_range,theta_slopes,alpha_range,output_folder,type):
    # 繪製不同角度的斜率圖
    plt.figure(figsize=(8, 6))
    theta_degrees = np.degrees(theta_range)
    match type:
        case 0:
            plt.plot(theta_degrees, theta_slopes)
            plt.xlabel('Theta (degrees)')
            plt.ylabel('Average Band Slope')
            plt.title('Average Band Slope vs. Theta')
            plt.grid(True)
            plt.savefig(f'{output_folder}/v vs theta/v_vs_theta_{theta_degrees[-1]:.2f}-{theta_degrees[0]:.2f}.png')
            plt.close()

        case 1:
            alpha_range = np.array(alpha_range)
            theta_slopes = np.log10(theta_slopes)
            minima_idx = argrelextrema(theta_slopes, np.less)[0]
            plt.plot(alpha_range, theta_slopes, marker='o')
            # 在相對極小值處標記
            plt.plot(alpha_range[minima_idx], theta_slopes[minima_idx], 'ro', label='Local Minima')
            # 標示相對極小值的座標
            for i in minima_idx:
                plt.text(alpha_range[i], theta_slopes[i], f'({alpha_range[i]:.4f}, {theta_slopes[i]:.2f})', fontsize=9, ha='right')
            plt.xlabel('Alpha')
            plt.ylabel('Average Band Slope')
            plt.title('Average Band Slope vs. Alpha')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_folder}/log(v) vs alpha/log(v)_vs_alpha_{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}.png')
            # plt.show()
            plt.close()

        case 2:
            plt.plot(alpha_range, theta_slopes)
            plt.xlabel('Alpha Square')
            plt.ylabel('Average Band Slope')
            plt.title('Average Band Slope vs. Alpha Square')
            plt.grid(True)
            plt.savefig(f'{output_folder}/v vs alpha/v_vs_alpha_{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}.png')
            plt.close()

        case 3:
            alpha_square = np.array(alpha_range) @ np.array(alpha_range)
            plt.plot(alpha_square, theta_slopes)
            plt.xlabel('Alpha Square')
            plt.ylabel('Average Band Slope')
            plt.title('Average Band Slope vs. Alpha Square')
            plt.grid(True)
            plt.savefig(f'{output_folder}/v vs alpha_square/v_vs_alpha_square_{alpha_square[0]:.2f}-{alpha_square[-1]:.2f}.png')
            plt.close()

def main():

    # 常數
    v, k_D, w = constants()
    s_0, s_1, s_2 = pauli_matrices()

    # Thetarange input
    m_range = np.arange(20,170) 
    theta_range = np.arccos((3*m_range**2+3*m_range+0.5)/(3*m_range**2+3*m_range+1))
    theta_slopes = []    # 保存不同角度的斜率25
    alpha_range = []
    
    # PLot velocity input
    plot_velocity_type = 2  #0:v vs theta 
                            #1:log(v/vf) vs alpha
                            #2:v vs alpha
                            #3:v vs alpha_square

    length_range = calculate_convergence_length(theta_range) # in the unit of 1/Anstrong
    length = length_range[-1]
    output_folder = 'Renormalized_Velocity'
    subfolders = ['v vs theta', 'log(v) vs alpha', 'v vs alpha','v vs alpha_square']
    setup_output_folder(output_folder, subfolders)
    print("The Output folder is "f'{output_folder}/{subfolders[plot_velocity_type]}')
    print(f'length is {length}')

    # Define Qpts .
    q_ = np.array([[0,1],[-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) #q_ is unit vector
    b_ = np.array([q_[0]-q_[1],q_[0]-q_[2]])
    Q_T = generate_Qpts(-q_[1], b_[0], b_[1], length)
    Q_B = generate_Qpts(np.zeros(2), b_[0], b_[1], length)
    Q = np.concatenate((Q_T, Q_B), axis=0)
    num_Qpts = Q.shape[0]
    band_indices = [num_Qpts-1,num_Qpts] #中間2條離0最近

    # Number of Kpoint
    num_kpts = 4    #1:Plot Charge 
                    #4:Plot Velocity
                    #20:Plot Band
    
    for theta in theta_range:
        k_theta, q_, alpha, b_ = theta_parameter(theta, v, k_D, w)
        Q_t = Q * k_theta 
        q_nn = np.array([q_[0],-q_[0],q_[1],-q_[1],q_[2],-q_[2]])

        k_paths = [                 # Path that used to calculate Velocity (Near B) and chrage_func
            (-0.1*q_[0], 0.1*q_[0])
        ]

        k_points = generate_kpts(k_paths, num_kpts)

        # Construct H
        energies , _ = calculate_band_structure(k_points, Q_t, q_nn, v, w, s_0, s_1, s_2,num_Qpts)

        # 計算能帶斜率平均
        slopes = calculate_band_slope(energies, k_points, band_indices)
        theta_slopes.append(slopes)
        alpha_range.append(alpha)

    theta_slopes = np.array(theta_slopes/v)

    df = pd.DataFrame({
    'm': m_range,
    'theta_range': theta_range,
    'alpha_range': alpha_range,
    'theta_slope': theta_slopes,
    })

    # 指定 Excel 文件路徑
    a = setup_output_folder('Dataframe')
    excel_path = f'{a}/velocity_{plot_velocity_type}.xlsx'

    # 使用 pandas 將 DataFrame 寫入 Excel 文件，不包含索引
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    plot_velocity(theta_range,theta_slopes,alpha_range,output_folder,plot_velocity_type) 



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} s")

        
# 建立 band class
# 計算函數: 有 calculate_band_structure, calculate_band_slope
# 畫圖函數: plot_band_structure, plot_velocity
# 變數: length, 常數, theta部分, Q部分