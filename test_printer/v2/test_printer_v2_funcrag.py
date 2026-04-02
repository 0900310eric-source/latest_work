import ollama
import pandas as pd
import numpy as np
import json
from pathlib import Path

MODELNAME = r"qwen2.5-coder:7b"
BASE_DIR = Path(__file__).parent
# C:\Users\eric0\Desktop\ML\eric_env\tools\test_printer\v2
local_test_lib_path = [str(BASE_DIR / 'numpy_lib_printer_v2.json')]

class FunctionRAG:
  def __init__(self, config_path):
     self.config_path = config_path

     with open(self.config_path, 'r', encoding='utf-8') as f:
       self.config = json .load(f)

  # 為防止 user_query 重複定義了已經在資料庫的函式, 建立搜索機制來避免
  def prompt_routing_agent(self, user_query):
    self.categories = []

    for item in self.config['rag_library']:
      info = f"- 類別: {item['intent']}\n  描述: {item['description']}\n  範例參考: {', '.join(item['examples'])}"
      # print(info)
      self.categories.append(info)

    # 函式資料庫
    knowledge_base = '\n'.join(self.categories)
    router_prompt = f"""
    你是一個語義匹配專家。請判斷使用者的需求最符合哪一個類別。
    
    這是你的知識庫：
    {knowledge_base}
    
    使用者目前的口頭需求："{user_query}"
    
    規則：
    1. 比對語義，而非字面。只要邏輯相似，就回傳類別名稱。
    2. 只回傳類別名稱（例如：threshold_filter），若無匹配則回傳 NONE。
    3. 嚴禁任何解釋。
    """
    response = ollama.generate(model='phi3', prompt=router_prompt)
    return response['response'].strip().lower().replace('.', '') # 將不必要的大小寫 一律改成小寫 & 消滅不必要的  -> ...
  
  # 使用 numpy_ib_printer_v2.json 的命名規則來建立 prompt
  def build_your_prompt(self, user_query):
    # 一開始就定義明確
    prompt = '你只是一個 numpy 專家. 現在根據使用者的要求, 生成 numpy 的函式, 生成方式請參考下方格式 產出 numpy 函式.'

    # 加入典型集範本 (Typical Set)
    prompt += "### 範本參考 ###\n"
    for lib in self.config['rag_library']:
      prompt += f"類別：{lib['description']}\n"
      prompt += f"代碼格式：\n{lib['code']}\n"
      prompt += "----------\n"

    # 加入使用者的實際任務
    prompt += f"\n### 實際任務 ###\n"
    prompt += f"使用者需求：{user_query}\n"
    prompt += """
              注意（必須嚴格遵守）：
              1. 函式名稱必須固定為: def logic(x, y):
              2. 參數 y 可能為 None。請務必在函式內第一步檢查 y，若 y 為 None 則根據 x 的形狀自行建立結果陣列 (例如 y = np.zeros_like(x) 或 np.copy(x))。
              3. 嚴禁使用 Python 的 for 迴圈。必須使用 Numpy 向量化操作（如切片 slicing、np.roll 等）以確保效能。
              4. 考慮到邊界問題：如果是差分運算，請確保處理後的陣列長度與輸入 x 一致。
              5. 只輸出 Python 代碼，不准有任何解釋文字或 Markdown 區塊。
              6. 函式必須回傳 (processed_x, processed_y)，其中第一個回傳值為橫軸數據，第二個為縱軸數據。
              """
    prompt += "代碼輸出："
    return prompt
  
  # x: test_data, y: result
  def render_job(self, test_data_x, test_data_y, user_query):
    # 檢查 user_query 是否有函式已經存在database
    # ****** 這是 user_query 的測試 ***********
    # user_query = "計算訊號 x 的 FFT 振幅頻譜。已知採樣頻率為 fs（由 y 傳入，若 y 為 None 則預設為 1000）。請產出雙邊頻譜轉換後的單邊振幅結果，並包含正確的頻率軸對應，確保輸出長度與輸入 x 一致。"
    selected_intent = self.prompt_routing_agent(user_query=user_query)

    raw_code = None
    found_in_db = False

    if selected_intent != 'none':
      for item in self.config['rag_library']:
        if selected_intent.strip() == item['intent'].lower():
          raw_code = item['code'].strip()
          found_in_db = True
          print(f"要求匹配成功：使用資料庫範本 [{selected_intent}]")
          break # 找到就停

    if not found_in_db:
      # prompt 後 得到 model 生成文字
      prompt = self.build_your_prompt(user_query=user_query)
      response = ollama.generate(model=MODELNAME, prompt= prompt)
      raw_code = response['response'].strip() # 拿掉多餘空格

    # 將生成 code 清乾淨
    clean_code = raw_code
    for junk in self.config['system_config']['clean_keywords']:
      clean_code = clean_code.replace(junk, '')
    clean_code = clean_code.strip()
  
    # 先叫好測試空間
    context = {'np': np}

    # 測試
    try:
      exec(clean_code, context)

      target_name = self.config['system_config']['target_function_name']
      logic_func = context.get(target_name)

      if logic_func:

        # --- 真正測試 --- ***** 做你函式的測試 *****

        # ****** 基準線測試, 做 FFT 基準測試 ********
        # fs = 1000
        # t = np.linspace(0, 0.1, num=100, endpoint=False)
        # f_test = 50
        # test_data = np.sin(2 * np.pi * f_test * t)
        # 這裡傳入 x=test_data, y=None
        # result = logic_func(test_data, fs)
        
        # ========= 匯入 test_printer_v2.py x, y array 資料  
        result = logic_func(test_data_x, test_data_y)    
        print(f"原始數據 x：{test_data_x}")
        print(f"原始數據 y:{test_data_y}")
        print(f"執行結果：{result}")
        # --------------------
        # 將錯誤尋找也錄進資料庫 (人工確認是否建立對的函式)
        if found_in_db:
          for item in self.config['rag_library']:
            if selected_intent.strip().lower() == item['intent'].strip().lower():
              print('\n' + '='*40)
              print(f"語意匹配成功, AI 匹配到這個函式類別 {item['intent']}")
              print(f"該函式描述: {item['description']}")
              print(f"執行程式預覽: {item['code']}")
              print('='*40)
              break

          # 人工確認是否為使用者喜歡
          feedback = input('\n這個邏輯是你想要的嗎？(y:正確 / n:這不是我要的，收錄為誤判): ')
          if feedback.lower() == 'y':
            return result[0], result[1]
          if feedback.lower() == 'n':
            for item in self.config['rag_library']:
              if selected_intent.strip().lower() == item['intent'].strip().lower():
                if 'bad_cases' not in item:
                  item['bad_cases'] = []

                # 先判斷 bad_cases 裡面有沒有符合的
                already_exist = False
                for obj in item['bad_cases']:
                  # 收錄 user_query 到 wrong_examples
                  if user_query in obj['wrong_examples']:
                    already_exist = True
                    break
                # 沒有,收錄
                if not already_exist:
                  new_bad_entry = {
                    "wrong_description": item['description'],
                    "wrong_examples":[
                      example for example in item['examples']
                    ] + [user_query],
                    "wrong_code": item['code']
                  }
                  item['bad_cases'].append(new_bad_entry)

            with open(self.config_path, 'w', encoding='utf-8') as f:
              json.dump(self.config, f, indent=4, ensure_ascii=False)
            return result[0], result[1]

        # --- 正向確認邏輯 (當 found_in_db 為 False 時) ---
        # 要先確認好新函式是不是自己要的再收錄
        if not found_in_db:
          print('\n' + '='*40)
          print("這是 AI 生成的新函式邏輯")
          print(f"使用者需求: {user_query}")
          print(f"新生成函式: {clean_code}")

          # 人工確認是否為使用者喜歡
          feedback = input('\n這個邏輯是你想要的嗎？(y:正確 / n:這不是我要的，收錄為誤判): ')

          if feedback == 'y':
            print("發現資料庫未收錄的技能，正在自動存入資料庫...")
        
            new_entry = {
              "intent": f"auto_{len(self.config['rag_library']) + 1}",
              "description": user_query,
              "examples": [user_query],
              "code": clean_code
            }
            # 寫入記憶體
            self.config['rag_library'].append(new_entry)
            # 寫回檔案 (持久化)
            with open(self.config_path, 'w', encoding='utf-8') as f:
              json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            print(f"學習成功！下次遇到類似需求將直接命中 [auto_{len(self.config['rag_library'])}]")
            return result[0], result[1]
          if feedback == 'n':
            return result[0], result[1]
      else:
        print("找不到 logic 函式，model 可能沒聽話改名")
  
    except Exception as e:
      print(f"執行出錯了：{e}")
      print(f"出錯的代碼內容是：\n{clean_code}")
    return

def main():

  # 建立我的 RAG
  functionrag = FunctionRAG(local_test_lib_path[0])
  functionrag.render_job("計算訊號 x 的 FFT 振幅頻譜。已知採樣頻率為 fs（由 y 傳入，若 y 為 None 則預設為 1000）。請產出雙邊頻譜轉換後的單邊振幅結果，並包含正確的頻率軸對應，確保輸出長度與輸入 x 一致。")

  # # 檢查 user_query 是否有函式已經存在database
  # user_query = "計算訊號 x 的 FFT 振幅頻譜。已知採樣頻率為 fs（由 y 傳入，若 y 為 None 則預設為 1000）。請產出雙邊頻譜轉換後的單邊振幅結果，並包含正確的頻率軸對應，確保輸出長度與輸入 x 一致。"
  # selected_intent = functionrag.prompt_routing_agent(user_query=user_query)

  # raw_code = None
  # found_in_db = False

  # if selected_intent != 'none':
  #   for item in functionrag.config['rag_library']:
  #     if selected_intent.strip() == item['intent'].lower():
  #       raw_code = item['code'].strip()
  #       found_in_db = True
  #       print(f"要求匹配成功：使用資料庫範本 [{selected_intent}]")
  #       break # 找到就停

  # if not found_in_db:
  #   # prompt 後 得到 model 生成文字
  #   prompt = functionrag.build_your_prompt(user_query=user_query)
  #   response = ollama.generate(model=MODELNAME, prompt= prompt)
  #   raw_code = response['response'].strip() # 拿掉多餘空格

  # # 將生成 code 清乾淨
  # clean_code = raw_code
  # for junk in functionrag.config['system_config']['clean_keywords']:
  #   clean_code = clean_code.replace(junk, '')
  # clean_code = clean_code.strip()
  
  # # 先叫好測試空間
  # context = {'np': np}

  # # 測試
  # try:
  #   exec(clean_code, context)

  #   target_name = functionrag.config['system_config']['target_function_name']
  #   logic_func = context.get(target_name)

  #   if logic_func:
  #     # --- 真正測試 --- ***** 做你函式的測試 *****
  #     # ****** 基準線測試, 做 FFT 基準測試 ********
  #     fs = 1000
  #     t = np.linspace(0, 0.1, num=100, endpoint=False)
  #     f_test = 50
  #     test_data = np.sin(2 * np.pi * f_test * t)
  #     # 這裡傳入 x=test_data, y=None
  #     result = logic_func(test_data, fs) 
            
  #     print(f"原始數據：{test_data}")
  #     print(f"執行結果：{result}")
  #     # --------------------
  #     # 將錯誤尋找也錄進資料庫 (人工確認是否建立對的函式)
  #     if found_in_db:
  #       for item in functionrag.config['rag_library']:
  #         if selected_intent.strip().lower() == item['intent'].strip().lower():
  #           print('\n' + '='*40)
  #           print(f"語意匹配成功, AI 匹配到這個函式類別 {item['intent']}")
  #           print(f"該函式描述: {item['description']}")
  #           print(f"執行程式預覽: {item['code']}")
  #           print('='*40)
  #           break

  #       # 人工確認是否為使用者喜歡
  #       feedback = input('\n這個邏輯是你想要的嗎？(y:正確 / n:這不是我要的，收錄為誤判): ')
  #       if feedback.lower() == 'n':
  #         for item in functionrag.config['rag_library']:
  #           if selected_intent.strip().lower() == item['intent'].strip().lower():
  #             if 'bad_cases' not in item:
  #               item['bad_cases'] = []

  #             # 先判斷 bad_cases 裡面有沒有符合的
  #             already_exist = False
  #             for obj in item['bad_cases']:
  #               # 收錄 user_query 到 wrong_examples
  #               if user_query in obj['wrong_examples']:
  #                already_exist = True
  #                break
  #             # 沒有,收錄
  #             if not already_exist:
  #               new_bad_entry = {
  #                   "wrong_description": item['description'],
  #                   "wrong_examples":[
  #                     example for example in item['examples']
  #                   ] + [user_query],
  #                   "wrong_code": item['code']
  #                 }
  #               item['bad_cases'].append(new_bad_entry)

  #         with open(functionrag.config_path, 'w', encoding='utf-8') as f:
  #             json.dump(functionrag.config, f, indent=4, ensure_ascii=False)

  #     # --- 正向確認邏輯 (當 found_in_db 為 False 時) ---
  #     # 要先確認好新函式是不是自己要的再收錄
  #     if not found_in_db:
  #       print('\n' + '='*40)
  #       print("這是 AI 生成的新函式邏輯")
  #       print(f"使用者需求: {user_query}")
  #       print(f"新生成函式: {clean_code}")

  #       # 人工確認是否為使用者喜歡
  #       feedback = input('\n這個邏輯是你想要的嗎？(y:正確 / n:這不是我要的，收錄為誤判): ')

  #       if feedback == 'y':
  #         print("發現資料庫未收錄的技能，正在自動存入資料庫...")
        
  #         new_entry = {
  #           "intent": f"auto_{len(functionrag.config['rag_library']) + 1}",
  #           "description": user_query,
  #           "examples": [user_query],
  #           "code": clean_code
  #         }
  #         # 寫入記憶體
  #         functionrag.config['rag_library'].append(new_entry)
  #         # 寫回檔案 (持久化)
  #         with open(functionrag.config_path, 'w', encoding='utf-8') as f:
  #             json.dump(functionrag.config, f, indent=4, ensure_ascii=False)
            
  #         print(f"學習成功！下次遇到類似需求將直接命中 [auto_{len(functionrag.config['rag_library'])}]")

  #   else:
  #     print("找不到 logic 函式，model 可能沒聽話改名")
  
  # except Exception as e:
  #   print(f"執行出錯了：{e}")
  #   print(f"出錯的代碼內容是：\n{clean_code}")

if __name__ == '__main__':
  main()