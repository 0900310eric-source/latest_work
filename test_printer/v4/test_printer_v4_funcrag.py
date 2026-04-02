import ollama
import json
import numpy as np
from pathlib import Path

MODELNAME = r"qwen2.5-coder:7b"
BASE_DIR = Path(__file__).parent
local_test_lib_path = [str(BASE_DIR / 'numpy_lib_printer_v4.json'), str(BASE_DIR / 'numpy_lib_printer_2_v4.json')]

class FunctionRAGV4:
  def __init__(self, config_path):
    self.config_path = config_path

    with open(self.config_path, 'r', encoding='utf-8') as f:
      self.config = json.load(f)

  # ============ 匹配 function 函式 =====================================================================================================
  def retrieve(self):
    
    # 搜尋結果 key: function_id, value: int
    results_function = {}
    max_value = 0 # ============================= 匹配的 weight 要大於0 (不能不相關或負相關) 才算中 ========================================
    # 不斷更新 best_doc 來找最匹配的 function
    best_doc = None
    length_lib = len(self.config['rag_library'])
    # 先將使用者輸入轉成小寫，比對時也把關鍵字轉小寫。
    # JSON 裡的 "FFT" 變成 "fft"，使用者的 "fft" 也是 "fft" -> 命中！
    query_lower = self.user_query.lower()  # <--- 1. 預先轉小寫

    for i, function in enumerate(self.config['rag_library']):
      # 先找那些 verified functions
      if function['status'] != 'verified':
        continue

      # 用權重搜索, 在 keywords (加最多) 內的加分, 在 negatives 的扣分
      weight = 0
      # 找 keywords (最重要)
      weight = weight + sum([0.1 for key in function['keywords'] if key.lower() in query_lower]) # 找把使用者命令小寫過的
      # weight = weight + sum([0.1 for description in function['description'] if description in self.user_query])
      weight = weight - sum([1 for negative in function['negatives'] if negative.lower() in query_lower]) # 找把使用者命令小寫過的
      results_function[function['function_id']] = weight

      # 搜索最有可能的結果
      if weight > max_value:
        max_value = weight
        best_doc = function

    # 將 best_doc (最相近結果回傳), 也有可能是 None
    if best_doc:
      print(f"RAG 命中: {best_doc['function_id']} (分數: {max_value})")
    else:
      print(f"RAG 未命中，將轉為純生成模式...")
    return best_doc

  # =============== 有找到匹配到就用這個 prompt ============================
  # 使用 numpy_ib_printer_v4.json 的命名規則來建立 prompt
  def build_rag_prompt(self, retrieved_doc):
    """
    將使用者查詢與檢索到的 JSON 資料組裝成 Prompt。
    
    Args:
        user_query (str): 使用者的中文問題 (e.g., "幫我把訊號做 50Hz 的低通")
        retrieved_doc (dict): 從 rag_library 中檢索到的那一筆字典
    
    Returns:
        str: 組裝好的完整 Prompt 字串
    """
    
    # 1. 處理 Description: 把 JSON 裡的字串陣列變成條列式文字
    # 這樣模型可以同時看到「數學定義」和「參數說明」
    description_text = "\n".join([f"- {line}" for line in retrieved_doc['description']])

    # 2. 組裝 Prompt (使用 f-string)
    prompt = f"""
    ### System Role ###
    You are a senior Python Data Scientist specializing in **NumPy** signal processing.
    Your goal is to write efficient, vectorized, and pure NumPy code.

    ### Strict Constraints ###
    1. **NO external libraries**: Do not import scipy, pandas, or matplotlib. Use `import numpy as np` only.
    2. **Input Flexibility**: The function `logic(x, y)` must handle `y` as a dictionary with default values (as shown in the reference).
    3. **Output Format**: Return ONLY the raw Python function text. Do NOT wrap it in Markdown. Do NOT add explanations.

    ---

    ### Reference Context (Golden Standard) ###
    We have retrieved a verified function that implies the correct logic.
    **Function ID:** {retrieved_doc['function_id']}

    **Logic Description (in Chinese):**
    {description_text}

    **Standard Code Pattern:**
    {retrieved_doc['code']}

    ---

    ### Current User Task ###
    **User Query (in Chinese):**
    "{self.user_query}"

    **Instruction:**
    Adapt the "Standard Code Pattern" above to satisfy the "User Query".
    - If the user specifies numeric values (e.g., "cutoff=50", "gain=3"), update the default values inside the `y.get(...)` lines.
    - Keep the logic structure identical to the reference.
    - **START YOUR RESPONSE DIRECTLY WITH:** `def logic(x, y):`

    ### Python Code:
    """
    return prompt
  
  # ================== 沒在資料庫找到就用這個 prompt ============================
  def build_gen_prompt(self):
    """
    當 RAG 找不到資料時，使用的純生成 Prompt。
    """
    prompt = f"""
    ### System Role ###
    You are a senior Python Data Scientist specializing in **NumPy**.
    
    ### Task ###
    Write a Python function named `logic(x, y)` based on the User Query.

    ### Strict Constraints (MUST FOLLOW) ###
    1. **Pure NumPy Only**: Use `import numpy as np`. NO scipy/pandas.
    2. **Function Signature**: `def logic(x, y):`
    3. **Parameter `x`**: Input numpy array.
    4. **Parameter `y`**: A configuration dictionary. You MUST handle default values using `y.get('param', default)`.
    5. **Output**: Return ONLY the python code. No markdown text, no comments.

    ### User Query ###
    "{self.user_query}"

    ### Instruction ###
    Think step-by-step about the math required.
    Implement the function `logic(x, y)` now.
    """
    return prompt
  
  # ================= 新增：驗證通過後，自動生成 Keywords 和 Description =================
  def generate_metadata(self, clean_code):
    """
    呼叫 LLM 閱讀最終確認的程式碼，並生成適合的 keywords 和 description。
    """
    print("正在為你的函式撰寫說明文件 (Keywords & Description)...")
    
    meta_prompt = f"""
    ### Task
    Analyze the following Python Function and User Query.
    Generate metadata for a RAG retrieval system.

    ### Input
    User Query: "{self.user_query}"
    Code:
    {clean_code}

    ### Output Requirements
    Return a STRICT JSON object with two keys:
    1. "keywords": A list of 3-5 English or Chinese keywords (related to the math/logic).
    2. "description": A list of 2 strings. Line 1 summarizes what it does. Line 2 explains the parameters (x, y).

    ### Example Output
    {{
        "keywords": ["linear algebra", "dot product", "內積"],
        "description": ["Calculates the dot product of two arrays.", "Parameter y implies the dimension axis."]
    }}

    ### JSON Response ONLY:
    """

    try:
        # 呼叫模型 (可以用比較快的小模型，這邊沿用原本的)
        response = ollama.chat(model=MODELNAME, messages=[{'role': 'user', 'content': meta_prompt}])
        content = response['message']['content']
        
        # 強制清理 Markdown 符號，防止模型回傳 ```json ... ```
        content = content.replace("```json", "").replace("```", "").strip()
        
        return json.loads(content) # 轉成 Python Dict 回傳
    except Exception as e:
        print(f"自動生成 Metadata 失敗，將使用預設空值: {e}")
        return {"keywords": ["auto_gen_failed"], "description": ["Description generation failed."]}
  
  # ============= 清理及生成新資料 ===============================================================================
  def verify_and_save_entry(self, clean_code):
    
    # 先看預覽畫面, 詢問是否正確
    print("="*40)
    print(clean_code)
    print("="*40)
    request = input("這是預覽程式碼, 請問這是你要的嗎? (y/n)\n ").lower()

    # 正確, 收錄進去
    if request == 'y':
      category = input("""這是哪一類: 1. math_linear [線性函數], 2. math_calc [微積分], 3. math_func [線性函數] (輸入 11-13)\n
                                     4. sig_filter [濾波器], 5. sig_trans [轉換], 6. sig_mod [調變], 7. sig_env [訊號特徵] (輸入 21-24)\n
                                     8. stat_basic [基礎統計], 9. stat_dist [機率分布], 10. stat_reg [曲線擬合], 11. stat_outlier [異常檢測], 12. stat_info [消息理論], 13. stat_test [統計檢定] (輸入 31-34)\n
                                     14. arr_shape [陣列重塑], 15. arr_mask [範圍限制], 16. arr_pad [填充處裡] (輸入 41-43)\n
                                     17. gen_wave [波形生成], 18. gen_noise [雜訊模擬] (輸入 51-52) \n""")
      intent_map = {'11': 'math_linear', '12': 'math_calc', '13': 'math_func',
                    '21': 'sig_filter', '22': 'sig_trans', '23': 'sig_mod', '24': 'sig_env',
                    '31': 'stat_basic', '32': 'stat_dist', '33': 'stat_reg', '34': 'stat_outlier',
                    '41': 'arr_shape', '42': 'arr_mask', '43': 'arr_pad',
                    '51': 'gen_wave', '52': 'gen_noise'}
      # category[0] 用來控制類別 1-5, category[1] 用來控制 函數種類 *******(目前同類別只有9種內)************
      if category in intent_map:
        # 拿到 keywords 和 description
        metadata = self.generate_metadata(clean_code)
        # 拿類別名
        prefix_name = intent_map[category]
        # 拿 index
        exist_count = 0
        for item in self.config['rag_library']:
          find = item.get('function_id', '')
          if find.startswith(prefix_name):
            exist_count += 1

        new_entry = {
          category[0]: f"{prefix_name}_",
          "function_id": f"{prefix_name}_{exist_count + 1}",
          "status": "pending",
          "keywords": metadata['keywords'],
          "description": metadata['description'],
          "code": clean_code,
          "negatives": []
        }
        self.config['rag_library'].append(new_entry)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    # 如果錯的話, 也許可以錄進negatives
    else:
      pass

  def render_job(self):

    # =========== 使用者輸入 =====================================================================================
    self.user_query = input("你的函式要求:\n ")

    # =========== 嘗試在資料庫中搜索 ==============================================================================
    retrieved_doc = self.retrieve()

    # =========== 生成 prompt (retrieved_doc == None 會改用 build_gen_prompt) ====================================
    if retrieved_doc:
      prompt = self.build_rag_prompt(retrieved_doc=retrieved_doc)
    else:
      prompt = self.build_gen_prompt()
    
    # =========== model 生成 ====================================================================================
    try:
      response = ollama.chat(model=MODELNAME, messages = [{'role': 'user', 'content': prompt}])
      # print("="*40)
      clean_code = (response['message']['content']).replace("```python", "").replace("```", "").strip() # 清理不重要地方
      # print(clean_code)
      # print("="*40)

      # ========= 清理及儲存 new_function ========================================================================
      self.verify_and_save_entry(clean_code)

    except Exception as e:
      print(f"發生錯誤: {e}")

def main():
  functionragv4 = FunctionRAGV4(local_test_lib_path[0])
  functionragv4.render_job()

if __name__ == '__main__':
  main()