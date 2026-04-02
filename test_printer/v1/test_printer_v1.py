import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from .test_printer_v1_logic import OperatorV1 as opv1
from .test_printer_v1_cols_mapping import ColumnMapping as ColMap

# test_printer_v1_parent abs address
BASE_DIR = Path(__file__).resolve().parent
# print(BASE_DIR) C:\Users\eric0\Desktop\ML\eric_env\tools\test_printer\v1

# 建立測試檔案位置
local_test_config_path = [str(BASE_DIR/"config_printer_v1.json")]
local_test_plot_path = [str(BASE_DIR/"figures_v1")]

class TestPrinterV1:
  def __init__(self, config_path, plot_path):
      self.config_path = config_path
      self.plot_path = Path(plot_path).resolve()
      self.plot_path.mkdir(parents=True, exist_ok=True)
  
  # var_name_list 從 x, y, z ...
  def load_files(self, group_name_list, var_name_list, prompt):
     # ***** 我的 agent 畫圖命令 *****
     self.prompt = prompt

     # 將 config.json 吃進來
     with open(self.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
     
     results = {}
     # project root
     # C:\Users\eric0\Desktop\ML\eric_env\tools\test_printer
     project_root = BASE_DIR.parent
     # print(project_root)

     for group in config['group']:
        # 檢查是否 group 在要求裡
        group_name = group['group_name']
        # 讓檔案
        label_list = [value for value in group['label']]
        level_list = [value for value in group['level']]
        if group_name in group_name_list:
           # 找到 group 路徑
           group_dir = (project_root / group['directory']).resolve()

           # 1. ***** 針對客製化 mapping *****
           colmap = ColMap()
           var_map = colmap.transition(var_name_list)
           
         #   # (old settings) 檢查檔案名字
         #   for dir in group_dir.iterdir():
         #   # 檢查是否為一個檔案
         #   # 檢查它的副檔名是 .csv
         #     if dir.is_file() and dir.suffix == '.csv':
         #       print(f"File name: {dir.name}")

           # 2. 打開目標檔案
           for file_path in group_dir.rglob('*.csv'):
               df = pd.read_csv(file_path)
               
               label = group.get('label', 'unknown')
               level = group.get('level', 'unknown')
               file_id = f"{group_name}_{label}_{level}"
               print(file_id)

               stats = df.loc[:, var_name_list].copy()
               stats = stats.rename(columns=var_map)
               # 3. 進入通用函式
               self.customized_operations(stats)
  
  # 通用邏輯函式
  def customized_operations(self, stats):
     pass
  
  # 這裡要加入畫圖
  
  def render_job(self):
     # 輸入想要變成 x, y...的欄位
     # 最後一欄是 prompt
     self.load_files(['group_1'], ['PlaybackTimeSec', 'BufferHealthSec'], 'y=2x')

def main():
  print("--- 程式開始執行 ---")
  try:
    # 檢查路徑是否存在
    config_p = local_test_config_path[0]
    plot_p = local_test_plot_path[0]
    print(f"計畫建立的路徑: {plot_p}")

    # 1. build object    
    testprintv1 = TestPrinterV1(config_p, plot_p)
    print("--- 實例化完成 ---")

    # 2. test run    
    testprintv1.render_job()
  except Exception as e:
    print(f"發生錯誤了: {e}")

if __name__ == '__main__':
  main()