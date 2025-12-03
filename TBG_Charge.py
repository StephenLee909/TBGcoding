import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import seaborn as sns
import time
from TBG_v1 import  setup_output_folder, \
                    constants,pauli_matrices,\
                    calculate_convergence_length,\
                    generate_Qpts,theta_parameter,\
                    calculate_band_structure

# class CHARGE:

#     v, k_D, w = constants()
#     s_0, s_1, s_2 = pauli_matrices()
#     q_ = np.array([[0,1],[-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) #q_ is unit vector
#     b_ = np.array([q_[0]-q_[1],q_[0]-q_[2]])
    
#     def __init__(self, m_range) -> None:

#         theta_range = np.arccos((3*m_range**2+3*m_range+0.5)/(3*m_range**2+3*m_range+1))
#         self.__Super_Lacttice_length_range = 2.464/(2*np.sin(theta_range/2))
#         length = calculate_convergence_length(theta_range) # in the unit of 1/Anstrong
#         # Define Qpts
        
#         Q_T = generate_Qpts(-q_[1], b_[0], b_[1], length)
#         Q_B = generate_Qpts(np.zeros(2), b_[0], b_[1], length)
#         Q = np.concatenate((Q_T, Q_B), axis=0)
#         num_Qpts = Q.shape[0]
#         band_indices = [num_Qpts-1,num_Qpts] #中間2條離0最近
#         self.charge_indicies = chrage_indicies(num_Qpts, num_QT)

# Calculate Charge  
def chrage_indicies(num_Qpts: int, num_QB: int) -> list[np.ndarray]:
    BA = np.arange(0, 2*num_QB, 2)
    BB = np.arange(1, 2*num_QB, 2)
    TA = np.arange(2*num_QB, 2*num_Qpts, 2)
    TB = np.arange(2*num_QB+1, 2*num_Qpts, 2)
    return [BA, BB, TA, TB]

def chrage_func(indicies, eigenvector, Q, r):
    z = 0
    for i in indicies:
        z += eigenvector[i] * np.exp(1j*(np.dot(Q[int(np.floor(i/2)),:],r)))
        # z += eigenvector[i]*np.conjugate(eigenvector[i]) test for vector normalization
    return np.abs(z)**2

def plot_charge(charge_indicies, band_indices, eigenvectors, Q, 
                output_folder, theta, Super_Lacttice_length):

    # 使用元組作為字典的鍵
    plot_indicies = [(0,),(2,),(0,1),(2,3),(0,2),(0,1,2,3),(1,),(3,),(1,3)]
    plot_indicies_name = ["BA","TA","BOT","TOP","A","TOT","BB","TB","B"]
    plot_indicies_dict = dict(zip(plot_indicies, plot_indicies_name))

    plot_k_points = [(0,),(1,),(2,),(0,1)]
    plot_k_points_name = ["K","K\'","G","K+K\'"]
    plot_k_points_dict = dict(zip(plot_k_points, plot_k_points_name))
    
    # 設置參數
    num_points = int(np.rint(Super_Lacttice_length))   # 網格點的數量
    bound = Super_Lacttice_length*2/3**0.5
    x = np.linspace(-bound, bound, num_points)      # x軸的範圍
    y = np.linspace(-bound, bound, num_points)      # y軸的範圍
    # 設定字型大小
    plt.rcParams['font.size'] = 14  # 設定全域的字型大小
    plt.rcParams['axes.titlesize'] = 20  # 設定標題的字型大小
    plt.rcParams['axes.labelsize'] = 20  # 設定坐標軸標籤的字型大小
    plt.rcParams['xtick.labelsize'] = 12  # 設定x軸刻度的字型大小
    plt.rcParams['ytick.labelsize'] = 12  # 設定y軸刻度的字型大小
    #邊緣
    plt.subplots_adjust(top=0.905,
                        bottom=0.145,
                        left=0.155,
                        right=0.945,
                        hspace=0.2,
                        wspace=0.2)
    # 生成網格點
    X, Y = np.meshgrid(x, y)

    # 初始化 Z 矩陣，將每個結果存入這裡以避免重複計算
    Z = np.zeros((X.shape[0], X.shape[1], 4, 4), dtype=float) #(x座標, y座標, 原子種類, Kpoint種類)
    
    # 計算每個網格點的Charge值並緩存結果
    for j in range(num_points):
        for k in range(num_points):
            r = np.array([X[j, k], Y[j, k]])
            for i in range(4):  # 4個原子種類 Modify here if you want to plot specific atom type
                for m in [0,1]:  # 3個k-point
                    Z[j, k, i, m] = np.sum([
                        chrage_func(charge_indicies[i], eigenvectors[m, :, l], Q, r) for l in band_indices
                    ])
                Z[j, k, i, 3] = Z[j, k, i, 0] + Z[j, k, i, 1]  # K+K'的結果

            # # 找出所有列不全為0的列索引
            # non_zero_columns = np.any(Z != 0, axis=2)
            # 移除所有值為0的列
    # Z = Z[:,:,np.any(Z != 0, axis=(0, 1, 2)),:]
    # print("remove 0 valued calumn")


    # 利用已計算的Z矩陣來生成圖
    for i in range(9):  # 9種原子排列組合
        for m in [0,1,3]:  # 4種k-point組合
        # for m in [2]:  # 4種k-point組合，Modify Here if you want to plot some specific Kpoints only
            Z_tot = np.zeros(X.shape, dtype=float)
            for index in plot_indicies[i]:  # 取出所需的原子組合
                Z_tot += Z[:, :, index, m]
            
            plt.figure(figsize=(8, 6))
            # 繪製等高線圖
            plt.contourf(X, Y, np.log10(Z_tot), levels=np.linspace(-5,1,80),extend='both', cmap='viridis')
            plot_charge_with_hexagon(Super_Lacttice_length)
            plt.colorbar(label='log(Charge Density)')
            plt.xlabel('X (Å)')
            plt.ylabel('Y (Å)')
            plot_type = plot_indicies_dict[tuple(plot_indicies[i])]
            K_point_type = plot_k_points_dict[tuple(plot_k_points[m])]
            plt.title(f'Charge distribution {plot_type} at {K_point_type}',)
            outputname =f'{output_folder}/chrage_func/{theta*180/np.pi:.4f}_{plot_type}_{K_point_type}.png'
            plt.savefig(outputname, dpi=800, bbox_inches='tight')
            print(f'Charge distribution {plot_type} at {K_point_type} has been plot')
            plt.close()

def plot_charge_with_hexagon(Super_Lacttice_length):
    L = Super_Lacttice_length/3**0.5
    # 定義六邊形的中心點和邊長
    hex_centers = np.array([[L, 0]])  # 可以根據需要調整中心點
    hex_radius = L  # 六邊形的邊長
    for center in hex_centers:
        hexagon = RegularPolygon(center, numVertices=6, radius=hex_radius, 
                                 orientation=np.pi/6,
                                 edgecolor='black', facecolor='none', linewidth=1, linestyle='--')
        plt.gca().add_patch(hexagon)

def main():

    # 常數
    v, k_D, w = constants()
    s_0, s_1, s_2 = pauli_matrices()

    # Thetarange input
    m_range = np.array([166])
    theta_range = np.arccos((3*m_range**2+3*m_range+0.5)/(3*m_range**2+3*m_range+1))
    Super_Lacttice_length_range = 2.464/(2*np.sin(theta_range/2))
    
    length = calculate_convergence_length(theta_range) # in the unit of 1/Anstrong
    output_folder = f'radius={length}'
    subfolders = ['chrage_func']
    setup_output_folder(output_folder, subfolders)
    print("The Output folder is "f'{output_folder}')

    # Define Qpts
    q_ = np.array([[0,1],[-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) #q_ is unit vector
    b_ = np.array([q_[0]-q_[1],q_[0]-q_[2]])
    Q_B = generate_Qpts(-q_[1], b_[0], b_[1], length)
    Q_T = generate_Qpts(np.zeros(2), b_[0], b_[1], length)
    Q = np.concatenate((Q_B, Q_T), axis=0)
    num_Qpts = Q.shape[0]
    band_indices = [num_Qpts-1,num_Qpts] #中間2條離0最近

    charge_indicies = chrage_indicies(num_Qpts, Q_B.shape[0])
    
    for i in range(len(theta_range)):
        k_theta, q_, alpha, b_ = theta_parameter(theta_range[i], v, k_D, w)
        Q_t = Q * k_theta 
        q_nn = np.array([q_[0],-q_[0],q_[1],-q_[1],q_[2],-q_[2]])

        k_points = np.array([[0,0],-q_[0],q_[2]]) # K point used to calculate chrage_func

        # Construct H
        print("Now Calculating the Charge")
        energies , eigenvectors = calculate_band_structure(k_points, Q_t, q_nn, v, w, s_0, s_1, s_2)

        # Calculate The chrage_func Density
        print("Now plotting the Charge")
        plot_charge(charge_indicies, band_indices, eigenvectors, Q_t, 
                    output_folder, theta_range[i], Super_Lacttice_length_range[i])


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} s")

        
# 建立 band class
# 計算函數: 有 calculate_band_structure, calculate_band_slope
# 畫圖函數: plot_band_structure, plot_velocity
# 變數: length, 常數, theta部分, Q部分