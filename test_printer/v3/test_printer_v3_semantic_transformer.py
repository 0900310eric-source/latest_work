import torch
import torch.nn as nn
from torch.nn import functional as fc
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).parent
local_test_lib_path = [str(BASE_DIR / 'numpy_lib_printer_v3.json')]

class SemanticTransformerV3(nn.Module):
  def __init__(self, conofig_path, len_words):
    super(SemanticTransformerV3, self).__init__()
    self.config_path = conofig_path

    # 建立 semantic space
    semantic_dim = 768
    self.semantic_space = nn.Embedding(len_words, semantic_dim)

    # ======= 載入預訓練模型 (這就是 BERT 的變體) =============
    # 1. 'all-MiniLM-L6-v2' 體積小(約80MB)、速度快，非常適合部署
    # 2. 這裡選擇 mpnet 模型，它會確保輸出是 768 維 (420MB)
    self.encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    with open(self.config_path, 'r', encoding='utf-8') as f:
      self.config = json.load(f)

  # 將文字轉入語意空間 (1, 768)
  def get_embedding(self, text):
    embedding = self.encoder.encode(text, convert_to_tensor=True)
    return embedding

  # ========== 將 numpy_lib_printer_v3.json 函式的 description 和 examples 轉成語意地圖 =====================
  def to_semantic_space(self):
    self.code_book = {}
    # 建立碼本
    for item in self.config['rag_library']:
      # 將同一 intent 進行打包
      local_typical_points = []

      # 將 description 和 examples 放入語意空間 (embedding dim = 768, ******* not one hot encoding *******)
      des_vec = self.get_embedding(item['description'])
      local_typical_points.append(des_vec)
      for example in item['examples']:
        example_vec = self.get_embedding(example)
        local_typical_points.append(example_vec)

      if item['intent'] not in self.code_book:
        # ========== 使用 torch.stack() 將 semantic vector 拼成 semantic map
        self.code_book[item['intent']] = torch.stack(local_typical_points)

  def search_semantic_map(self, user_query, threshold=0.6):
    # 將要搜索文字座標化
    user_code = self.get_embedding(user_query)

    scores = {}
    # 搜索相似性
    for intent, intent_matrix in self.code_book.items():
      # 要先將一維 vector 擴充成二維
      # fc.cosine_similarity 可以直接 [1, 768] 和 [N, 768] 比相似性
      score = fc.cosine_similarity(user_code.unsqueeze(0), intent_matrix)

      # 抓每個 intent 內 (description + examples) 裡面相似最高的
      scores[intent] = torch.max(score).item()

    max_similarity = max([similarity for _, similarity in scores.items()])
    selected_intents = [intent for intent, similarity in scores.items() if similarity == max_similarity]

    if max_similarity <= threshold:
      print(f"語意相似度太低, 您的 user_query 可能沒有匹配到任何 intent")
      return None, max_similarity
    return selected_intents, max_similarity

  def render_job(self):

    # 1. 先把語意轉至語意空間
    self.to_semantic_space()

    # 2. 將 user_query 進入語意空間搜索
    while True:
      user_query = input("\n [Your query]: ").lower().strip()
      if user_query in ['e', 'q', '離開']:
        break

      threshold = 0.5
      selected_intents, confidence = self.search_semantic_map(user_query=user_query, threshold=threshold)

      if selected_intents != None:
        # 只檢查最高的, 高維空間幾乎不會撞值
        target_intent = selected_intents[0]

        for item in self.config['rag_library']:
           if target_intent == item['intent']:
              print(f"[AI助手]: 偵測匹配 intent: {target_intent},   信心度: {confidence}")
              print("-" * 30)
              print(f"程式為: \n{item['code']}")
              print("-" * 30)
      else:
          print("[AI 助手]: 抱歉，我不太理解您的意思。您可以試著說「泰勒展開」或「矩陣乘法」。")

def main():
  # 多少字在這語意空間
  len_words = 1000

  semantranv3 = SemanticTransformerV3(local_test_lib_path[0], len_words)
  semantranv3.render_job()

if __name__ == '__main__':
  main()