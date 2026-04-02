import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from .test_printer_v2_cols_mapping import ColumnMapping as ColMap
from .test_printer_v2_logic import OperatorV2 as opv2
from .test_printer_v2_funcrag import FunctionRAG as funcrag

# test_printer_v1_parent abs address
BASE_DIR = Path(__file__).resolve().parent
# C:\Users\eric0\Desktop\ML\eric_env\tools\test_printer\v2
# ========= funcrag json ===================
local_test_lib_path = [str(BASE_DIR / 'numpy_lib_printer_v2.json')]

# 建立測試檔案位置
local_test_config_path = [str(BASE_DIR/"config_printer_v2.json")]
local_test_plot_path = [str(BASE_DIR/"figures_v2")]

class TestPrinterV2:
  def __init__(self, config_path, plot_path):
    self.config_path = config_path
    self.plot_path = Path(plot_path).resolve()
    self.plot_path.mkdir(parents=True, exist_ok=True)

  def search_and_match_files(self, file_path):
     relative_dir_path = file_path.relative_to(BASE_DIR)
     # 找到相對位置才能改檔名
     relative_file_path = relative_dir_path.relative_to(self.config['settings']['input_dir'])
     names = str(relative_file_path.stem).split('_')
     print(names)

     # 找檔案是否配到 mapping table
     id = ''
     for name in names:
         if name in self.config['mapping']['group']:
            id = id + name + '_'
            continue
         if id != '' and name in self.config['mapping']['label']:
            id = id + name + '_'
         if id != '' and name in self.config['mapping']['level']:
            id = id + name
     raw_id = '_'.join([value for value in id.split('_')])
     raw_file_path = f"{raw_id}{file_path.suffix}"
    # print(raw_file_path)   

      # 改檔案名並且丟到 raw_dir (絕對路徑) ***** 先建立 raw_dir *****
     raw_dir = BASE_DIR / self.config['settings']['raw_dir']
     raw_dir.mkdir(parents=True, exist_ok=True)
     target_file_path = raw_dir / raw_file_path
     file_path.rename(target_file_path)
    # print(file_path)

  def make_mask(self, mask_list, group_name_list, label_name_list, level_name_list):
     self.mask_list = mask_list
     self.group_name_list = group_name_list
     self.label_name_list = label_name_list
     self.level_name_list = level_name_list

     with open(self.config_path, 'r', encoding='utf-8') as f:
        self.config = config = json.load(f)
        input_dir = BASE_DIR/self.config['settings']['input_dir']

     # build mapping table
     for idx, mask in enumerate(self.mask_list):
        if idx == 0:
           for group_name in self.group_name_list:
               if mask:
                  self.config['mapping']['group'][group_name] = True
               else:
                  self.config['mapping']['group'][group_name] = False
        elif idx == 1:
           for label_name in self.label_name_list:
               if mask:
                  self.config['mapping']['label'][label_name] = True
               else:
                  self.config['mapping']['label'][label_name] = False
        elif idx == 2:
           for level_name in self.level_name_list:
               if mask:
                  self.config['mapping']['level'][level_name] = True
               else:
                  self.config['mapping']['level'][level_name] = False

     print(self.config['mapping']['group'])
     print(self.config['mapping']['label'])
     print(self.config['mapping']['level'])

     # 寫回去 json
     with open(self.config_path, 'w', encoding='utf-8') as f:
            # indent=4 可以讓 JSON 縮排，看起來比較漂亮
            # ensure_ascii=False 確保中文不會變成亂碼
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        
     print(f"Mapping 已成功寫入 {self.config_path}")
     
     for file_path in input_dir.rglob('*.csv'):
        self.search_and_match_files(file_path)
  
#   def load_and_parse_mask_json(self, mask_list, group_name_list, label_name_list, level_name_list):
#      with open(self.config_path, 'r', encoding='utf-8') as f:
#         config = json.load(f)

#      # 決定到底要不要畫(group: on/off)_(label: on/off)_(level: on/off)
#      # on: 1, off: 0
#      for idx, mask in enumerate(mask_list):
#         if mask:
#            config['mapping_mask'] = 1
#         else:
#            config['mapping_mask'] = 0
  
  def load_files(self, var_list):
     colmap = ColMap()
     varmap = colmap.transition(var_list)
     self.results = {}
     
     # ***** 資料從 input_dir -> raw_dir 之後, 要重新去找 raw_dir 因為 input_dir 已空 *****
     raw_dir = BASE_DIR / self.config['settings']['raw_dir']
     for file in raw_dir.rglob('*.csv'):
        raw_file_path = file.relative_to(raw_dir)
        temp_id = str(raw_file_path.stem).split('_')
        raw_id = '_'.join([value for value in temp_id])
        # print(raw_id)
        # print(raw_file_path)

        df = pd.read_csv(file)
        stats = df.loc[:, var_list].copy()
        stats = stats.rename(columns=varmap)
       #   print(stats)
     return stats

  def customized_operations(self, stats, user_query):
   #   # 1. 把 self.x 做 FFT
   #   f_test = 55
   #   t = np.linspace(0, 0.2, num=2000, endpoint=False)
   #   self.x = np.sin(2 * np.pi * f_test * t)
   #   self.y = 10000

   #   # ========= 把 funcrag 匯入 ===============
   #   func = funcrag(local_test_lib_path[0])
   #   processed_x, processed_y = func.render_job(self.x, self.y, user_query)

      # 2. # --- 手動測試泰勒展開 ---
           # 建立橫軸 x: 從 -pi 到 pi
     self.x = np.linspace(-np.pi, np.pi, 1000)
     self.y = 5
     func = funcrag(local_test_lib_path[0])
     processed_x, processed_y = func.render_job(self.x, self.y, user_query)
     real_sin = np.sin(self.x)
     error = np.max(np.abs(processed_y - real_sin))
     print(f"error: {error}")
     return processed_x, processed_y
  
  def render_job(self):
     # mask list 未來會做成勾選的, 用戶自己輸入group names, label names, level names
     self.make_mask([1, 1, 1], ['Group1'], ['VR', 'Music', 'News', 'Sports', 'Movies', 'Animation'], ['200', '20'])

     # ========== user_query ===========
     user_query = "我想算出 sin 的近似，不要太複雜的，大概 3 階就好。"
     stats = self.load_files(['PlaybackTimeSec', 'BufferHealthSec'])
     # self.load_and_parse_mask_json([0, 0, 0], ['group_1'], ['VR', 'Music', 'News', 'Sports', 'Movies', 'Animation'], ['200', '20'])

     # 進入通用函式
     processed_x, processed_y = self.customized_operations(stats, user_query=user_query)
     # --- 開始設計 plt ---
     plt.figure(figsize=(12, 6.75))
      
     # 畫出處理後的結果
     plt.plot(processed_x, processed_y, label='Processed Result', color='orange', linewidth=1.5)
      
     # 修飾圖表
   # ======== 1. FFT ==============
   #   plt.xlabel("Frequency (Hz)")
   #   plt.ylabel("FFT amplitude")
   #   plt.grid(True, linestyle='--', alpha=0.7)
   #   plt.legend()

   # ======== 2. Tylor expansion =========
     plt.xlabel("x")
     plt.ylabel("sin(x)")
     plt.grid(True, linestyle='--', alpha=0.7)
     plt.legend()
      
     # 存檔或顯示
     save_path = self.plot_path / "test_printer_v2_test1.png"
     plt.savefig(save_path)
     print(f"圖表已儲存至: {save_path}")
     plt.show()
     

def main():
   testprinterv2 = TestPrinterV2(local_test_config_path[0], local_test_plot_path[0])
   testprinterv2.render_job()

if __name__ == '__main__':
   main()