import numpy as np
import time

# 生成一個隨機的四維陣列
arr = np.random.rand(500, 500, 4, 4)

# 指定沿著軸 2 和軸 3 的部分索引範圍
indices_2 = np.array([1, 3])  # 例如，索引 1 和 3
indices_3 = np.array([0, 2])  # 例如，索引 1 和 3

# 方法 1: 先提取子陣列，再求和
start_time = time.time()
sub_array = arr[:, :, :, indices_3]
sub_array = sub_array[:, :, indices_2, :]
sum_sub_array = np.sum(np.sum(sub_array, axis=3), axis=2)  # 沿著軸 3 和 2 求和
time_sub_array = time.time() - start_time
print(f"提取子陣列再求和的時間: {time_sub_array:.6f} 秒")

# 方法 2: 使用 for 迴圈來直接計算
start_time = time.time()
sum_for_loop = np.zeros((arr.shape[0], arr.shape[1]))
for idx2 in indices_2:
    for idx3 in indices_3:
        sum_for_loop += arr[:, :, idx2, idx3]
time_for_loop = time.time() - start_time
print(f"使用 for 迴圈直接計算的時間: {time_for_loop:.6f} 秒")

# 驗證兩種方法的結果是否相同
assert np.allclose(sum_sub_array, sum_for_loop), "結果不一致"

# 結果顯示
if time_sub_array < time_for_loop:
    print("先提取子陣列再求和的方法較快。")
else:
    print("使用 for 迴圈直接計算的方法較快。")

'''
提取子陣列再求和的時間: 0.019567 秒
使用 for 迴圈直接計算的時間: 0.012007 秒
使用 for 迴圈直接計算的方法較快。
'''