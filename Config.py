# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
task_name = 'BTXRD' # Hoặc MoNuSeg, Covid19
learning_rate = 3e-4
batch_size = 64

##########################################################################
# Semi-Supervised Learning configs (UPDATED)
##########################################################################
semi_supervised = False
labeled_ratio = 0.5
beta = 0.99

warmup_epochs = 50       # 50 epoch đầu chỉ chạy Supervised
rampup_length = 200      # Độ dài giai đoạn tăng trưởng trọng số
lambda_max = 0.3         # Trọng số tối đa cho pseudo-loss

# -----------------------------

model_name = 'LViT'
train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

# ... (Giữ nguyên phần get_CTranS_config và test_session)
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 1
    return config

test_session = "BioBERT_ACI_02.21_13h19"