
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

# resolved_archive_file = "../user_data/pretrain_model/bert-base-chinese/pytorch_model.bin"
# resolved_archive_file = "../user_data/output_model/crf/pytorch_model.bin"
# state_dict = torch.load(resolved_archive_file,map_location="cpu")
# print("Models state_dict:")
# for param_tensor in state_dict:
#     print(param_tensor, "\t", state_dict[param_tensor].size())
#     if "transitions" in param_tensor:
#         print(param_tensor, "\t", state_dict[param_tensor])

pred = torch.tensor([[0, 1, 2, 1, 2, 2],
                     [2, 1, 2, 0, 1, 2]])
pred_np = pred.clone()
logits = torch.tensor(np.arange(36).reshape([2,6,3]))

print(pred)
print(logits)

batch_size = pred.size(0)
seq_len = pred.size(1)

# for i in range(batch_size):
#     start_index = -1
#     end_index = -1
#     for j in range(seq_len):
#         # print("pred_np[{}][{}] {}".format(i, j, pred_np[i][j]) )
#         # print("start_index ", start_index)
#         # print("end_index ", end_index)
#         if pred_np[i][j] == 0:
#             if start_index != -1 and end_index != -1:
#                 logits[i, start_index:end_index, :] = torch.sum(logits[i, start_index:end_index, :], dim=0,
#                                                                 keepdim=True).repeat(end_index - start_index, 1)
#             start_index = -1
#             end_index = -1
#         elif pred_np[i][j] == 1:
#             if start_index != -1 and end_index != -1:
#                 logits[i, start_index:end_index, :] = torch.sum(logits[i, start_index:end_index, :], dim=0,
#                                                                 keepdim=True).repeat(end_index - start_index, 1)
#             start_index = j
#             end_index = j+1
#         else:
#             end_index = j+1
#
#         if j == seq_len - 1 and pred_np[i][j] == 2:
#             if start_index != -1 and end_index != -1:
#                 logits[i, start_index:end_index, :] = torch.sum(logits[i, start_index:end_index, :], dim=0,
#                                                                 keepdim=True).repeat(end_index - start_index, 1)

# segment_logits_list = []
# for i in range(batch_size):
#     acc_index = 0
#     pred_np[i][0] = 0
#     for j in range(1, seq_len):
#         if pred_np[i][j] != 2:
#             acc_index += 1
#         pred_np[i][j] = acc_index
#
#     segment_logits_list.append(torch.zeros_like(logits[i]).index_add_(0, pred_np[i], logits[i]))
# segment_logits = torch.concat(segment_logits_list, dim=0).view(batch_size, seq_len, -1)
# segment_logits = torch.gather(segment_logits, 1, pred_np.view(batch_size,seq_len,1).expand(batch_size,seq_len,3))
#
# print(pred_np)
# print(segment_logits)








# mask_idx = torch.tensor([0,1,1,1,0]).bool()
# mask_idx = mask_idx.view(5, 1).expand(5, 3)
# cur_partition = torch.arange(15).reshape(5,3)
# masked_cur_partition = cur_partition.masked_select(mask_idx).float()
# mask_idx = mask_idx.contiguous().view(5, 3, 1)
# partition = torch.zeros(5,3,1)
# partition.masked_scatter_(mask_idx, masked_cur_partition)
# print(masked_cur_partition)
# print(partition)


print(torch.__version__)

