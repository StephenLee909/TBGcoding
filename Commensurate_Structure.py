import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import seaborn as sns
from TBG_v1 import setup_output_folder

class Reciprocal_Space:

    def __init__(self, point=np.array([4*np.pi/(3*2.464),0])) -> None:
        self.point = point
        self.hex = self.generate_Hexagon
        self.b_ = np.linalg.norm(self.point) * 3**0.5 * np.array([[-3**0.5/2,0.5],[3**0.5/2,0.5]])
    
    @property
    def generate_Hexagon(self):
        hex_array = np.zeros((6,2))
        hex_array[0]=self.point
        for i in range(1,6):
            hex_array[i]= Reciprocal_Space.Rot @ hex_array[i-1]
        return hex_array
    
    @property
    def Plot(self):
        # Add Hexagon Axes
        plt.figure(figsize=(16, 9))
        polygon = plt.Polygon(self.hex, closed=True, color='skyblue')
        plt.scatter(self.hex[:, 0], self.hex[:, 1], color='blue', marker='o')
        plt.gca().add_patch(polygon)

        # Add Reciprocal Lattice
        X = np.array([0, 0])
        Y = np.array([0, 0])
        U = self.b_[:,0]
        V = self.b_[:,1]
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.005,color='blue')
        
        # 添加標籤
        labels = ['b2', 'b1']
        for i in range(len(X)):
            plt.text(X[i] + U[i], Y[i] + V[i], labels[i], fontsize=35, color='black', ha='right')
        plt.text(0, 0, "Γ", fontsize=60, color='black', ha='right')
        plt.text(self.hex[0, 0], self.hex[0, 1], "K", fontsize=60, color='black', ha='right')
        plt.text(self.hex[3, 0], self.hex[3, 1], "K\'", fontsize=60, color='black', ha='right')
        plt.text(np.average(self.hex[0:2, 0]), np.average(self.hex[0:2, 1]), "M", fontsize=60, color='black', ha='right')
        
        # Axis 
        plt.xlabel('kx (1/Å)', fontsize = 35)
        plt.ylabel('ky (1/Å)', fontsize = 35)
        plt.title(f'Reciprocal Space', fontsize = 35)
        # 調整座標軸數字標籤的字體大小
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlim(-10,4)
        plt.ylim(-3,3)
        plt.grid(True)
        plt.axis('equal')
        
        # Output
        setup_output_folder('reciprocal_space')
        outputname = 'reciprocal_space/Reciprocal_Lattice.png'
        plt.savefig(outputname, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    @staticmethod
    def R(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    Rot = R(np.pi/3)


def generate_Qpts(vector, b0, b1, length):
    """生成 Q 點網格"""
    vectors_set = set()
    for i in range(-length, length):
        for j in range(-length, length):
            vectors_set.add(tuple(vector + i*b0 + j*b1))
    return np.array(list(vectors_set))


def generate_Rotation_Lattice(theta, length):
    """生成旋轉晶格點"""
    R = Reciprocal_Space.R(theta)
    a = 2.46
    a_ = a * np.array([[-0.5, 3**0.5 / 2], [0.5, 3**0.5 / 2]])
    t_ = np.array([np.zeros(2), (a_[0] + a_[1]) / 3])
    
    P_1 = generate_Qpts(R @ t_[0], R @ a_[0], R @ a_[1], length)
    P_2 = generate_Qpts(R @ t_[1], R @ a_[0], R @ a_[1], length)
    P = np.concatenate((P_1, P_2), axis=0)
    
    return P


def get_parameters():
    """
    獲取使用者輸入的參數
    返回: (m, length, bound)
    """
    print("="*50)
    print("🎯 Moiré Pattern Generator")
    print("="*50)
    print("\n📋 輸入參數說明:")
    print("  • m: 扭轉晶格參數 (預設: 15)")
    print("  • length: 晶格搜尋範圍 (預設: 50)")
    print("  • bound: 圖形顯示範圍 (預設: 60)")
    print("\n💡 直接按 Enter 使用預設值\n")
    
    # 輸入 m
    m_input = input("請輸入 m 值 [預設: 15]: ").strip()
    try:
        m = int(m_input) if m_input else 15
    except ValueError:
        print(f"❌ 無效輸入，使用預設值 m=15")
        m = 15
    
    # 輸入 length
    length_input = input("請輸入 length 值 [預設: 50]: ").strip()
    try:
        length = int(length_input) if length_input else 50
    except ValueError:
        print(f"❌ 無效輸入，使用預設值 length=50")
        length = 50
    
    # 輸入 bound
    bound_input = input("請輸入 bound 值 [預設: 60]: ").strip()
    try:
        bound = int(bound_input) if bound_input else 60
    except ValueError:
        print(f"❌ 無效輸入，使用預設值 bound=60")
        bound = 60
    
    print("\n" + "="*50)
    print(f"✅ 參數設定完成: m={m}, length={length}, bound={bound}")
    print("="*50 + "\n")
    
    return m, length, bound


def main():
    """主函數 - 輸入 m 參數，輸出 Moiré 圖案"""
    
    # 獲取使用者輸入參數
    m, length, bound = get_parameters()
    
    # 建立輸出資料夾
    output_folder = "Moiré Pattern"
    subfolders = ["plot_points"]
    setup_output_folder(output_folder, subfolders)
    
    # 晶格常數
    a = 2.46
    a_ = a * np.array([[-0.5, 3**0.5 / 2], [0.5, 3**0.5 / 2]])
    t_ = np.array([np.zeros(2), (a_[0] + a_[1]) / 3])
    
    print(f"⚙️  計算 m={m} 的 Moiré 圖案...\n")
    
    # 計算扭轉角
    theta = np.arccos((3*m**2 + 3*m + 0.5) / (3*m**2 + 3*m + 1))
    print(f"  扭轉角 θ = {np.degrees(theta):.4f}°")
    
    # 生成兩層晶格點
    print(f"  生成晶格點 (length={length})...")
    P = generate_Rotation_Lattice(theta/2, length)      # 上層（+θ/2）
    Q = generate_Rotation_Lattice(-theta/2, length)     # 下層（-θ/2）
    
    print(f"  上層點數: {P.shape[0]}")
    print(f"  下層點數: {Q.shape[0]}")
    
    # 計算 moiré 晶格參數
    t1 = m * a_[0] + (m + 1) * a_[1]
    t1 = Reciprocal_Space.R(theta) @ t1
    moiré_size = np.linalg.norm(t1)
    print(f"  Moiré 晶格大小: {moiré_size:.4f} Å\n")
    
    # 繪製 Moiré 圖案
    print(f"📈 繪製圖案 (bound={bound})...")
    plt.figure(figsize=(10, 10), dpi=150)
    
    # 散點圖
    plt.scatter(P[:, 0], P[:, 1], color='blue', marker='o', s=1, label='Layer 1 (+θ/2)', alpha=0.7)
    plt.scatter(Q[:, 0], Q[:, 1], color='red', marker='o', s=1, label='Layer 2 (-θ/2)', alpha=0.7)
    
    # # 添加 moiré 晶格圓圈
    # circle = plt.Circle((0, 0), moiré_size, color='black', fill=False, linestyle='--', linewidth=2, label='Moiré lattice')
    # plt.gca().add_artist(circle)
    
    # 標籤和設定
    # plt.title(f'm = {m}, θ = {np.degrees(theta):.4f}°', fontsize=20, fontweight='bold')
    plt.xlabel('x (Å)', fontsize=16)
    plt.ylabel('y (Å)', fontsize=16)
    # plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 設定座標範圍
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    
    # 儲存與顯示
    filename = f'{output_folder}/plot_points/m_{m}_theta_{np.degrees(theta):.2f}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✅ 圖片已保存: {filename}\n")
    
    plt.show()
    plt.close()
    
    print("✨ 執行完成！")
    return 0


if __name__ == "__main__":
    main()