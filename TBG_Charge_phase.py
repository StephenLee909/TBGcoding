import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import seaborn as sns
import time
import pandas as pd
from typing import Tuple
from TBG_v1 import setup_output_folder,\
                    constants,\
                    pauli_matrices,\
                    generate_Qpts,\
                    calculate_convergence_length,\
                    hamiltonian
from TBG_Charge import chrage_indicies

def alpha_parameters(alpha_range: np.ndarray, v: float, k_D: float, w: float, num_path: int) -> \
                    Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: #Not Commensurate 
    k_theta_range = w/(v*alpha_range)
    theta_range = 2*np.arcsin(k_theta_range/(2*k_D))
    q_ = np.array([[0, 1], [-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) * k_theta_range[:,np.newaxis,np.newaxis]
    q_nn = np.concatenate([q_[:, 0:1, :], #In order To calculate the right Hamiltonian
                           -q_[:, 0:1, :],
                           q_[:, 1:2, :], 
                           -q_[:, 1:2, :],
                           q_[:, 2:3, :], 
                           -q_[:, 2:3, :]], axis=1)

    L = 2.464/(2*np.sin(theta_range/2)*3**0.5)
    path = np.linspace(-1,2,num_path) * L[:,np.newaxis]
    return theta_range, k_theta_range, q_nn, path

def Q_parameters(length: int, k_theta_range: np.ndarray) -> Tuple[np.ndarray, list[int], np.ndarray]:
    q0_ = np.array([[0,1],[-3**0.5 / 2, -0.5], [3**0.5 / 2, -0.5]]) #q0_ is unit vector
    b0_ = np.array([q0_[0]-q0_[1],q0_[0]-q0_[2]])
    Q_B = generate_Qpts(-q0_[1], b0_[0], b0_[1], length)
    Q_T = generate_Qpts(np.zeros(2), b0_[0], b0_[1], length)
    Q0 = np.concatenate((Q_B, Q_T), axis=0)
    band_indices = [Q0.shape[0]-1,Q0.shape[0]] #中間2條離0最近
    charge_indicies = chrage_indicies(Q0.shape[0], Q_B.shape[0])
    Q = Q0 * k_theta_range[:,np.newaxis,np.newaxis]
    return Q, band_indices, charge_indicies

# 計算能帶結構
def calculate_band_structure(theta_range: np.ndarray, k_points: np.ndarray, 
                             Q: np.ndarray, q_nn: np.ndarray, 
                             v: float, w: float, s_0: np.ndarray, s_1: np.ndarray, s_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print("Calculating the Band")
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

def chrage_func(indicies: np.ndarray, eigenvector: np.ndarray, Q: np.ndarray, r: float):
    z = 0
    for i in indicies:
        z += eigenvector[i] * np.exp(1j*(Q[int(np.floor(i/2)),0]*r))
        # z += eigenvector[i]*np.conjugate(eigenvector[i]) 
        # test for vector normalization
    return np.abs(z)**2

def calculate_partial_data(alpha_i: float, diff=0.5, num_pts = 50, num_path = 50, outputfolder = "Dataframe"):
    print(f'Now Calculate Partial data, alpha_i = {alpha_i}')
    starttime=time.time()

    # 常數
    v, k_D, w = constants()
    s_0, s_1, s_2 = pauli_matrices()

    # Alpha Input
    alpha_range = np.linspace(alpha_i, alpha_i+diff, num_pts)  # alpha等差 
    theta_range, k_theta_range, q_nn, path = alpha_parameters(alpha_range, v, k_D, w, num_path)

    # Length Modify
    length = calculate_convergence_length(theta_range[-1:]) # in the unit of 1/Anstrong

    # Define Qpts
    Q, band_indices, charge_indicies = Q_parameters(length,k_theta_range)

    # Calculate Hamiltonian
    k_points = np.array([[0,0],[-1,0],[3**0.5/2,-0.5]]) * k_theta_range[:, np.newaxis, np.newaxis] # K point used to calculate chrage_func

    # Construct H (input 0*s_0 as CSCM)
    _ , eigenvectors = calculate_band_structure(theta_range, k_points, Q, q_nn, v, w, s_0, s_1, s_2)
    
    # 初始化 Z 矩陣，將每個結果存入這裡以避免重複計算
    Z = np.zeros((num_path, num_pts, 4, 4), dtype=float) #(phase, alpha, 原子種類, Kpoint種類)

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
    print(f'End Calculating, time: {time.time() - starttime:.2f} s')
    print("Start Writing Partial Result")
    starttime=time.time()

    # 創建一個 ExcelWriter 物件來儲存數據
    output_file = f'{setup_output_folder(outputfolder)}/{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        for i in range(4):
            for j in range(4):
                # 取得 Z[:,:,i,j] 這個 2D 部分數據
                partial_data = Z[:, :, i, j]
                # 將部分數據轉換為 DataFrame 來儲存到 Excel
                df = pd.DataFrame(partial_data)
                sheet_name = f'Z_{i}_{j}'
                # 將 DataFrame 儲存到對應的工作表中
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'End Writing, time: {time.time() - starttime:.2f} s')

def partial_data_join(outputfolder = "Dataframe"):
    print("Start Joining Partial Result")
    starttime=time.time()

    # 定義輸入檔案和輸出檔案的名稱
    input_files = []
    for i in range(6):
        input_files.append(f'{setup_output_folder(outputfolder)}/{0.5+i*0.5:.2f}-{1.0+i*0.5:.2f}.xlsx')
    output_file = f'{setup_output_folder(outputfolder)}/0.50-3.50.xlsx'

    # 初始化一個字典來存儲拼接後的數據
    concatenated_data = {}

    # 讀取每個輸入檔案並提取數據
    for file in input_files:
        xls = pd.ExcelFile(file)
        for sheet_name in xls.sheet_names:
            # 讀取每個工作表的數據
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            # 如果該工作表的數據已存在於字典中，則進行拼接
            if sheet_name in concatenated_data:
                concatenated_data[sheet_name] = pd.concat([concatenated_data[sheet_name], df], axis=1)
            else:
                concatenated_data[sheet_name] = df

    # 創建一個新的 ExcelWriter 來保存拼接後的數據
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data in concatenated_data.items():
            # 將拼接後的數據寫入新的工作表中
            data.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    print(f'End Writing Output file at {output_file}\n time: {time.time() - starttime:.2f} s')

def plot_charge_phase_from_df(num_pts = 300, num_path = 50):
    print("Start Ploting Charge from Datafram")
    starttime=time.time()

    # 讀取total的Excel檔案
    total_file = f'{setup_output_folder("Dataframe")}/0.50-3.50.xlsx'
    # 逐個讀取每個工作表
    for sheet_name in pd.ExcelFile(total_file).sheet_names:
        # 讀取工作表，保留標題行（header=0）
        df = pd.read_excel(total_file, sheet_name=sheet_name, header=0)
        Z = df.values
        # 生成x和y軸的範圍
        X, Y = np.meshgrid(np.linspace(0.5,3.50,num_pts), np.linspace(-1,2,num_path))

        # 創建圖形
        plt.figure(figsize=(8, 6))
        
        # 繪製等高線圖
        plt.contourf(X, Y, np.log10(Z), levels=np.linspace(-3,1,61), extend='both', cmap='viridis')
        plt.colorbar(label='log(Charge Density)')
        plt.xlabel('alpha')
        phase_labels = ['AB', 'AA', 'BA', 'AB']
        phase_indices = [-1, 0, 1, 2]
        plt.yticks(phase_indices, phase_labels)

        plt.title(f'Charge Phase {sheet_name}')
        # 標示magic angle degree
        for j in [0.6047, 1.2788, 1.8142, 2.7064, 3.3012]:
            plt.axvline(x=j, color="grey", linestyle="--")
        plt.title(f'Contour plot for {sheet_name}')
        
        # 保存圖形
        plt.savefig(f'Charge_phase/BM/plot_{sheet_name}.png', dpi=300)
        plt.close()
    print(f'End Writing Output file at Charge_phase/BM/\ntime: {time.time() - starttime:.2f} s')

def load_excel_to_dict(alpha_i, diff, input_file = "Dataframe") -> dict:
    print("Start Loading Excel to numpy array")
    starttime=time.time()
    # 初始化一個字典來儲存所有工作表的數據
    Z_dict = {}
    file_path = f'{input_file}/{alpha_i:.2f}-{alpha_i+diff:.2f}.xlsx'
    # 讀取 Excel 檔案
    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            # 假設工作表名格式為 "Z_i_j"，例如 "Z_0_0"
            parts = sheet_name.split('_')
            i = int(parts[1])
            j = int(parts[2])
            
            # 讀取數據並存儲到字典中
            Z_dict[(i, j)] = pd.read_excel(xls, sheet_name=sheet_name, header=0).to_numpy()

    print(f'End Loading, time: {time.time() - starttime:.2f} s')
    return Z_dict

def plot_charge_phase_from_dict(Z_dict: dict,alpha_i , diff, num_pts= 300, num_path = 50):
    print("Now Plotting Charge Phase from dict")
    starttime=time.time()
    # 設定字型大小
    plt.rcParams['font.size'] = 14  # 設定全域的字型大小
    plt.rcParams['axes.titlesize'] = 18  # 設定標題的字型大小
    plt.rcParams['axes.labelsize'] = 18  # 設定坐標軸標籤的字型大小
    plt.rcParams['xtick.labelsize'] = 14  # 設定x軸刻度的字型大小
    plt.rcParams['ytick.labelsize'] = 14  # 設定y軸刻度的字型大小
    #邊緣
    plt.subplots_adjust(top=0.899,
                        bottom=0.165,
                        left=0.09,
                        right=0.934,
                        hspace=0.2,
                        wspace=0.2)
    #Plot indicies & K points
    plot_indicies = [(0,), (2,), (0,1), (2,3), (0,2), (0,1,2,3), (1,), (3,), (1,3)]
    plot_indicies_name = ["BA", "TA", "BOT", "TOP", "A", "TOT", "BB", "TB", "B"]
    plot_indicies_dict = dict(zip(plot_indicies, plot_indicies_name))
    plot_k_points = [(0,), (1,), (2,), (0,1)]
    plot_k_points_name = ["K", "K'", "G", "K+K'"]
    plot_k_points_dict = dict(zip(plot_k_points, plot_k_points_name))

    # Plot Phase
    X, Y = np.meshgrid(np.linspace(alpha_i,alpha_i+diff,num_pts), np.linspace(-1, 2, num_path))

    # Outputfile
    

    for i in plot_indicies:  # 6種原子排列組合
        plot_type = plot_indicies_dict[tuple(int(x) for x in i)]
        for m in plot_k_points:    # 4種k-point組合
            K_point_type = plot_k_points_dict[tuple(int(x) for x in m)]
            Z_tot = np.zeros((X.shape[0], X.shape[1]), dtype=float)
            for index in i:  # 取出所需的原子組合
                for index_k in m:
                    Z_tot += Z_dict[(index, index_k)]

            # 繪製等高線圖
            plt.contourf(X, Y, np.log10(Z_tot), np.linspace(-5,1,81), extend='both', cmap='viridis')
            plt.colorbar(label='log(Charge Density)',format='%.1f')
            plt.xlabel('alpha')
            phase_labels = ['AB', 'AA', 'BA', 'AB']
            phase_indices = [-1, 0, 1, 2]
            plt.yticks(phase_indices, phase_labels)
            plt.title(f'Charge Phase {plot_type} at {K_point_type}')
            # 標示magic angle degree
            for j in [0.6047, 1.2788, 1.8142, 2.7064, 3.3012]:
                plt.axvline(x=j, color="black", linestyle="--", linewidth=2)
            outputname = f'{setup_output_folder(f'Charge_phase/BM/{alpha_i:.2f}-{alpha_i+diff:.2f}')}/{alpha_i:.2f}-{alpha_i+diff:.2f}_{plot_type}_{K_point_type}.png'
            plt.savefig(outputname, dpi=800)
            plt.show()
            plt.tight_layout()
            plt.close()
    
    print(f'End Ploting, time: {time.time() - starttime:.0f} s')

def plot_charge_phase(Z: np.ndarray, output_folder: str, alpha_range: np.ndarray, num_path: int):
# 使用元組作為字典的鍵
    print("Now Plotting Charge Phase")
    plot_indicies = [(0,),(2,),(0,1),(2,3),(0,2),(0,1,2,3),(1,),(3,),(1,3)]
    plot_indicies_name = ["BA","TA","BOT","TOP","A","TOT","BB","TB","B"]
    plot_indicies_dict = dict(zip(plot_indicies, plot_indicies_name))

    plot_k_points = [(0,),(1,),(2,),(0,1)]
    plot_k_points_name = ["K","K\'","G","K+K\'"]
    plot_k_points_dict = dict(zip(plot_k_points, plot_k_points_name))

        #Plot Phase
    X, Y = np.meshgrid(alpha_range, np.linspace(-1,2,num_path))

    # 利用已計算的Z矩陣來生成圖
    for i in plot_indicies[2]:  # 6種原子排列組合
        plot_type = plot_indicies_dict[tuple(int(x) for x in plot_indicies[i])]
        for m in [1,3]:  # 4種k-point組合
            K_point_type = plot_k_points_dict[tuple(plot_k_points[m])]
            Z_tot = np.zeros((X.shape[0], X.shape[1]), dtype=float)
            for index in plot_indicies[i]:  # 取出所需的原子組合
                Z_tot += Z[:, :, index, m]
            
            # 繪製等高線圖
            plt.contourf(X, Y, np.log10(Z_tot), levels=np.linspace(-5,1,20),extend='both', cmap='viridis')
            plt.colorbar(label='log(Charge Density)')
            plt.xlabel('alpha')
            phase_labels = ['AB', 'AA', 'BA', 'AB']
            phase_indices = [-1, 0, 1, 2]
            plt.yticks(phase_indices, phase_labels)
            plt.title(f'Charge Phase {plot_type} at {K_point_type}')
            # 標示magic angle degree
            for j in [0.6047, 1.2788, 1.8142, 2.7064, 3.3012]:
                plt.axvline(x=j, color="grey", linestyle="--")
            outputname = f'{output_folder}/BM/{alpha_range[0]:.2f}-{alpha_range[-1]:.2f}_{plot_type}_{K_point_type}.png'
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

def main():

    # #Output Partial Result:
    # for alpha in np.arange(0.5,2.0,0.5):
    #     calculate_partial_data(alpha)
    # #Join Partial Result:
    # partial_data_join()
    num_pts = 200
    num_path = 200
    alpha_i = 0.5
    diff = 3
    # calculate_partial_data(alpha_i, diff, num_pts, num_path,"Dataframe/test")
    #Plot Charge Phase
    Z_dict = load_excel_to_dict(alpha_i, diff,"Dataframe/test")
    plot_charge_phase_from_dict(Z_dict, alpha_i, diff, num_pts, num_path)


    # plot_charge_phase_from_df(num_pts, num_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} s")

        
# 建立 band class
# 計算函數: 有 calculate_band_structure, calculate_band_slope
# 畫圖函數: plot_band_structure, plot_velocity
# 變數: length, 常數, theta部分, Q部分