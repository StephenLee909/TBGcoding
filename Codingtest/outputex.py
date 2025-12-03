import numpy as np
import pandas as pd
import os 
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

# 生成一個隨機的四維陣列 (phase, alpha, 原子種類, Kpoint種類)
X_shape = 100
Y_shape = 100
Z = np.random.rand(X_shape, Y_shape, 4, 4)


# 創建一個 ExcelWriter 物件來儲存數據
output_file = f'{setup_output_folder("Dataframe")}/partial_results.xlsx'
output_file = '1.xlsx'
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
            

print(f"已將部分結果儲存到 {output_file}")

# 定義輸入檔案和輸出檔案的名稱
input_files = ['0.xlsx', '1.xlsx']
output_file = '0-1.xlsx'

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

print(f"已將結果儲存到 {output_file}")
