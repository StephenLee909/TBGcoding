import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import seaborn as sns
from matplotlib.patches import RegularPolygon
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
        plt.axis('equal')  # 保證 x 與 y 軸相同比例
        #Output
        setup_output_folder('reciprocal_space')
        outputname = 'reciprocal_space/Reciprocal_Lattice.png'
        plt.savefig(outputname, bbox_inches='tight',dpi=300)

    @staticmethod
    def R(theta):
        return np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
    Rot = R(np.pi/3)


# Q point
def generate_Qpts(vector,b0,b1,length):
    vectors_set = set()
    for i in range(-length,length):
        for j in range(-length,length):
                vectors_set.add(tuple(vector+i*b0+j*b1))
    return np.array(list(vectors_set))

# def plot_point(points, output_folder, Title, length):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(points[:, 0], points[:, 1], color='blue', marker='o')
    
#     # 為每個點添加標籤（如果需要）
#     # for i, point in enumerate(points):
#     #     plt.text(point[0], point[1], str(i), fontsize=9, ha='right')
    
#     plt.xlabel('X coordinate')
#     plt.ylabel('Y coordinate')
#     plt.title(f'{Title}_# of total pts = {points.shape[0]}')
#     plt.grid(True)
#     plt.axis('equal')  # 保證 x 與 y 軸相同比例
#     # plt.show()

#     # 繪製圓
#     circle = plt.Circle((0, 0), length, color='red', fill=False, linestyle='--')
#     plt.gca().add_artist(circle)
    
#     filename = f'{output_folder}/plot_points/{Title}.png'
#     plt.savefig(filename)
#     plt.close()

def generate_Rotation_Lattice(theta, length):
    R=Reciprocal_Space.R(theta)
    a = 2.46
    a_ = a * np.array([[-0.5, 3**0.5 / 2], [0.5, 3**0.5 / 2]])
    t_ = np.array([np.zeros(2),(a_[0]+a_[1])/3])
    P_1 = generate_Qpts(R@t_[0], R@a_[0], R@a_[1], length)
    P_2 = generate_Qpts(R@t_[1], R@a_[0], R@a_[1], length)
    P = np.concatenate((P_1, P_2), axis=0)
    return P_1




def main():
    
    R_space = Reciprocal_Space()
    R_space.Plot



    return 0
    output_folder = "Moiré Pattern"
    subfolders = ["plot_points"]
    setup_output_folder(output_folder, subfolders)
    
    a = 2.46
    a_ = a * np.array([[-0.5, 3**0.5 / 2], [0.5, 3**0.5 / 2]])
    t_ = np.array([np.zeros(2),(a_[0]+a_[1])/3])    
    length = 50
    # m = 10
    for m in range(15,16):
        theta = np.arccos((3*m**2+3*m+0.5)/(3*m**2+3*m+1))
        P = generate_Rotation_Lattice(theta/2, length)
        Q = generate_Rotation_Lattice(-theta/2, length)
        # P = generate_Rotation_Lattice(0, length)
        # Q = generate_Rotation_Lattice(-theta, length)


        # S = generate_Rotation_Lattice(0, length)
        X = generate_Rotation_Lattice(0+np.pi/3, length)
        Y = generate_Rotation_Lattice(2*theta+np.pi/3, length)
        t1 = m*a_[0]+(m+1)*a_[1]
        t1 = R(theta) @ t1
        C = []
        for i in range(P.shape[0]):
            if np.abs(np.linalg.norm(P[i,:]) - 13**0.5)<= 0.001:
                C.append(P[i,:])
                print(i)
        C = np.array(C)
        print(C.shape[0])
    
        # plot_point(P, output_folder,"Moiré Pattern",length)
        plt.figure(figsize=(8, 8))
        plt.title(f'm = {m}, theta = {np.degrees(theta):.4f}°')
        plt.scatter(P[:, 0], P[:, 1], color='blue', marker='o',s = 10)
        # plt.scatter(C[:, 0], C[:, 1], color='purple', marker='^',s = 30)
        plt.scatter(Q[:, 0], Q[:, 1], color='red', marker='o',s = 10)
        plt.xlabel('x(Å)')
        plt.ylabel('y(Å)')
        # plt.scatter(t1[0], t1[1], color='black', marker='o',s = 10)
        # plt.scatter(X[:, 0], X[:, 1], color='orange', marker='o',s = 10)
        # plt.scatter(Y[:, 0], Y[:, 1], color='cyan', marker='o',s = 10)
        # plt.scatter(S[:, 0], S[:, 1], color='black', marker='o',s = 15)
        # circle = plt.Circle((0, 0), (13**0.5), color='black', fill=False, linestyle='--')
        circle = plt.Circle((0, 0), (np.linalg.norm(t1)), color='black', fill=False, linestyle='--')
        # circle1 = plt.Circle((0, 0), (np.linalg.norm(t1))*7**0.5, color='black', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        # plt.gca().add_artist(circle1)


        # plt.axline((0, 0), slope=3**0.5, color='black', linestyle="--")
        # plt.axline((0, 0), slope=-3**0.5, color='black', linestyle="--")
        # plt.axhline(y=0, color="black", linestyle="--")
        # plt.axline((0, 0), slope=3**-0.5, color='grey', linestyle="--")
        # plt.axline((0, 0), slope=-3**-0.5, color='grey', linestyle="--")
        # plt.axvline(x=0, color="grey", linestyle="--")

        # plt.axline((0, 0), slope=np.tan(-theta/2), color='black', linestyle="--")
        # plt.axline((0, 0), slope=np.tan(np.pi/3-theta/2), color='black', linestyle="--")
        # plt.axline((0, 0), slope=np.tan(2*np.pi/3-theta/2), color='black', linestyle="--")
        # plt.axline((0, 0), slope=np.tan(np.pi/6-theta/2), color='grey', linestyle="--")
        # plt.axline((0, 0), slope=np.tan(np.pi/6+np.pi/3-theta/2), color='grey', linestyle="--")
        # plt.axline((0, 0), slope=np.tan(np.pi/6+2*np.pi/3-theta/2), color='grey', linestyle="--")
        # plt.axline((0, 0), slope=-3**0.5, color='grey', linestyle="-")

        # plt.grid(True)
        plt.axis('equal')  # 保證 x 與 y 軸相同比例
        bound = 60
        plt.xlim(-bound,bound)
        plt.ylim(-bound,bound)
        plt.savefig(f'{output_folder}/plot_points/m = {m}_theta = {np.degrees(theta):.2f}.png', dpi = 800)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()

        
# 建立 band class
# 計算函數: 有 calculate_band_structure, calculate_band_slope
# 畫圖函數: plot_band_structure, plot_velocity
# 變數: length, 常數, theta部分, Q部分