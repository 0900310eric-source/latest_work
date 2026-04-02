import torch
import torch.nn.functional as F
import torch.nn as nn

logits = torch.tensor([
    [2.0, 0.5, -1.0],
    [0.1, 0.2, 0.7],
])  # (2, 3)

target_index = torch.tensor([0, 2])  # (2,)
target_onehot = F.one_hot(target_index, num_classes = 3).float()

log_probs = F.log_softmax(logits, dim = -1)
per_sample_loss = - (target_onehot * log_probs).sum(dim=-1)
loss_one_hot = per_sample_loss.mean()
print(loss_one_hot)
# ce = nn.CrossEntropyLoss()
# loss_index = ce(logits, target_index)
# print(loss_index)