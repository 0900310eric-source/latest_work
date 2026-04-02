class ColumnMapping:
  def __init__(self):
    pass
  
  # 讓每次 mapping 都可以重新刷新
  def transition(self, var_name_list, var_number=2):
    mapping_name_order_list = ['x', 'y']
    mapping = {}

    if len(var_name_list) != var_number:
      print(f"請重新選擇變數數量, 只能兩個")

    for idx, var_name in enumerate(var_name_list):
        mapping[var_name] = mapping_name_order_list[idx]

    return mapping