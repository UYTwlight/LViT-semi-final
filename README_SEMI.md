# Hướng dẫn Training Semi-Supervised LViT

## Tổng quan

Code này triển khai **Semi-Supervised Learning** cho LViT với:
- **EPI (Exponential Pseudo-label Iteration)**: Tự học từ dữ liệu không nhãn
- **LV Loss (Language-Vision Loss)**: Học qua chỉ dẫn văn bản

## Cơ chế hoạt động

### 1. EPI (Exponential Pseudo-label Iteration)
```
P_t = β·P_{t-1} + (1-β)·P_current
```
- β = 0.99 (ema_decay)
- Tạo nhãn giả ổn định từ EMA model
- Nhãn giả được làm mượt qua từng step

### 2. LV Loss (Language-Vision Loss)
```
L_LV = 1 - ImgSim(pred, contrastive_label)
```
- Tìm ảnh có văn bản tương đồng (Text Similarity)
- Mượn nhãn đối chiếu từ ảnh đó
- Giám sát: Càng giống nhãn đối chiếu càng tốt

### 3. Training Loop
```
L_total = L_sup + L_unsup + α_LV·L_LV
```
- **L_sup**: Loss từ 25% dữ liệu có nhãn
- **L_unsup**: Loss từ 75% dữ liệu không nhãn + pseudo labels
- **L_LV**: Text-guided contrastive loss

## Cách chạy

### 1. Chạy Semi-Supervised Training
```bash
python train_model_semi.py
```

### 2. Tham số cấu hình (Config.py)
```python
batch_size = 2          # Batch size cho labeled data
epochs = 2000           # Số epochs
learning_rate = 1e-3    # Learning rate
model_name = 'LViT'     # Model type
```

### 3. Cấu trúc dữ liệu yêu cầu

```
datasets/MoNuSeg/
├── Train_Folder/
│   ├── img/              # Ảnh có nhãn (25%)
│   ├── labelcol/         # Nhãn tương ứng + dữ liệu không nhãn (75%)
│   └── Train_text.xlsx   # Mô tả văn bản
├── Val_Folder/
│   ├── img/
│   ├── labelcol/
│   └── Val_text.xlsx
└── Test_Folder/
    ├── img/
    ├── labelcol/
```

## Các file chính

1. **train_model_semi.py**: Main training script
   - Tạo labeled & unlabeled dataloaders
   - Khởi tạo model & EMA model
   - Training loop với early stopping

2. **Train_one_epoch_semi.py**: Core training logic
   - `train_one_epoch_semi()`: Main training function
   - `update_ema_variables()`: EMA update (EPI)
   - `compute_lv_loss()`: LV Loss computation
   - `print_summary_semi()`: Print training progress

3. **Load_Dataset.py**: Dataset classes
   - `ImageToImage2D`: Labeled data (có ảnh + nhãn)
   - `LV2D`: Unlabeled data (chỉ có nhãn/masks)

4. **utils.py**: Loss functions
   - `WeightedDiceBCE`: Supervised loss
   - `WeightedDiceBCE_unsup`: Unsupervised loss (có LV loss)
   - `img_similarity_vectors_via_numpy`: Image similarity

## Hyperparameters quan trọng

```python
ema_decay = 0.99              # EMA decay rate (β)
batch_size = 2                # Labeled batch size
unlabeled_batch_size = 6      # Unlabeled batch size (3x)
dice_weight = 0.5             # Dice loss weight
BCE_weight = 0.5              # BCE loss weight
LV_weight = 0.1               # LV loss weight (trong utils.py)
```

## Output

```
MoNuSeg/LViT/Test_session_XX.XX_XXhXX/
├── models/
│   └── best_model-LViT.pth.tar
├── tensorboard_logs/
│   └── events.out.tfevents...
├── visualize_val/
│   ├── 10/
│   ├── 20/
│   └── ...
└── Test_session_XX.XX_XXhXX.log
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=MoNuSeg/LViT/Test_session_XX.XX_XXhXX/tensorboard_logs/
```

### Log metrics
- Train_Total_Loss
- Train_Sup_Loss (L_sup)
- Train_Unsup_Loss (L_unsup)
- Train_LV_Loss (L_LV)
- Train_IoU
- Train_Dice

## Ưu điểm

✅ Học từ **ít dữ liệu có nhãn** (25%)
✅ Tận dụng **nhiều dữ liệu không nhãn** (75%)
✅ **EPI** tạo nhãn giả ổn định
✅ **LV Loss** dùng văn bản để kiểm tra tính hợp lý
✅ Đạt hiệu suất cao với ít nhãn

## Lưu ý

1. Đảm bảo có file `Train_text.xlsx` với cột `Image` và `Description`
2. Cấu trúc thư mục phải đúng như trên
3. Unlabeled data được load từ `labelcol/` folder
4. EMA model tự động update mỗi step
5. Best model được lưu khi val_dice tăng

## Troubleshooting

**Q: Out of memory?**
- Giảm `batch_size` trong Config.py
- Giảm unlabeled batch size (sửa `batch_size * 3` -> `batch_size * 2`)

**Q: LV Loss rất cao?**
- Kiểm tra text embeddings có đúng shape không
- Đảm bảo có đủ labeled data trong bank

**Q: Pseudo labels không tốt?**
- Tăng `ema_decay` (0.99 -> 0.995)
- Train supervised trước vài epochs

## Citation

```bibtex
@article{LViT-Semi,
  title={Semi-Supervised Learning for LViT with EPI and LV Loss},
  author={Your Name},
  year={2024}
}
```
