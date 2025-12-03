import numpy as np
import pandas as pd
from TBG_v1 import constants, setup_output_folder

def calculate_convergence_length(theta_range):
    est_Qnum = 5/np.degrees(theta_range)**2
    est_length = (est_Qnum*3**1.5/np.pi)**0.5
    return est_length


# 計算數據
m_range = np.arange(1, 300, 1)
theta_range = np.arccos((3*m_range**2 + 3*m_range + 0.5) / (3*m_range**2 + 3*m_range + 1))
v, k_D, w = constants()
k_theta = 2 * k_D * np.sin(theta_range / 2)
alpha_range = w / (v * k_theta)
length = calculate_convergence_length(theta_range)
Morieperiod =   2.464/(2*np.sin(theta_range/2))


# 創建 DataFrame
df = pd.DataFrame({
    'm': m_range,
    'theta_rad': theta_range,
    'theta_deg': np.degrees(theta_range),
    'alpha': alpha_range,
    'alpha_square': alpha_range**2,
    'length':length,
    'Morie_period':Morieperiod
})

# 指定 Excel 文件路徑
a = setup_output_folder('Dataframe')
excel_path = f'{a}/m_theta_alpha_length.xlsx'

# 使用 pandas 將 DataFrame 寫入 Excel 文件，不包含索引
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

    # 獲取工作表
    worksheet = writer.sheets['Sheet1']

    # 設定各列的格式
    for idx, col in enumerate(df.columns, 1):
        if col == 'm':
            for cell in worksheet.iter_cols(min_col=idx, max_col=idx, min_row=1, max_row=len(df) + 1):
                cell[0].number_format = '0'  # 整數格式
        else:
            for cell in worksheet.iter_cols(min_col=idx, max_col=idx, min_row=1, max_row=len(df) + 1):
                cell[0].number_format = '0.000E+00'  # 科學記號格式

print(f"Excel file '{excel_path}' has been created.")
