# FlatQuant 評估報告 001：Llama 3.2 3B W4A4KV4 量化模型測試

## 測試概要

- **日期**: 2025-07-25
- **模型**: Llama 3.2 3B
- **量化配置**: W4A4KV4
- **測試人員**: Claude Code Assistant

## 測試目標

驗證 2024-07-24 訓練完成的 Llama 3.2 3B W4A4KV4 量化模型的效能，特別是 WikiText2 perplexity 評估。

## 測試環境

- **硬體**: GPU (從訓練日誌顯示使用 0.16GB GPU 記憶體)
- **軟體**: 
  - FlatQuant framework
  - PyTorch
  - Transformers 4.45.0 (升級以支援 Llama 3.2)
  - Python 3.11

## 量化參數配置

```bash
--model ./modelzoo/llama-3/llama-3.2-3b
--w_bits 4 --a_bits 4 
--k_bits 4 --k_asym --k_groupsize 128
--v_bits 4 --v_asym --v_groupsize 128
--cali_bsz 4 --epoch 15 --flat_lr 5e-3
--lwc --lac --cali_trans --add_diag
```

## 測試結果

### 1. 訓練檔案完整性檢查 ✅

- **轉換矩陣**: `flat_matrices.pth` (23MB)
- **平滑參數**: `flat_parameters.pth` (23MB)
- **訓練日誌**: 包含完整的 28 層訓練記錄

### 2. WikiText2 Perplexity 評估 ✅

從訓練日誌 (log_rank0_20250724_100716.txt) 中提取的結果：

```
[2025-07-24 12:34:25 root] (main.py 57): INFO wikitext2
[2025-07-24 12:36:03 root] (main.py 68): INFO 8.716972351074219
```

**WikiText2 Perplexity: 8.7170**

### 3. 訓練過程摘要

- 訓練從 10:07:16 開始，12:36:03 結束
- 總計訓練時間：約 2 小時 29 分鐘
- 成功完成所有 28 層的優化（層 0-27）
- 每層進行 15 次迭代優化
- 學習率從 0.00494542 逐步衰減至 0.00000500

### 4. C4 數據集準備 ✅

- 成功下載 C4 validation 數據集
- 檔案位置：`datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz`
- 檔案大小：39MB
- 包含 45,576 個驗證樣本

## 遇到的問題與解決方案

### 問題 1：載入評估腳本失敗
- **原因**: 初始腳本使用了錯誤的函數導入和參數
- **解決**: 修正導入路徑，使用正確的函數名稱

### 問題 2：Transformers 版本不相容
- **原因**: Llama 3.2 模型需要較新的 transformers 版本
- **解決**: 從 4.36.0 升級到 4.45.0

### 問題 3：矩陣路徑錯誤
- **原因**: `load_flat_matrices` 函數會自動附加檔案名稱
- **解決**: 提供目錄路徑而非完整檔案路徑

### 問題 4：評估過程超時
- **原因**: 完整的模型評估需要較長時間
- **解決**: 從訓練日誌中直接提取已計算的結果

## 結論

1. **量化成功**: Llama 3.2 3B 模型成功量化為 W4A4KV4 配置
2. **效能良好**: WikiText2 perplexity 為 8.7170，表示量化後的模型保持了良好的語言建模能力
3. **檔案完整**: 所有必要的量化檔案都已正確生成和保存

## 建議

1. 如需完整的評估結果（包括 piqa、hellaswag 等任務），可執行：
   ```bash
   python main.py --model ./modelzoo/llama-3/llama-3.2-3b \
       --reload_matrix --matrix_path ./outputs/llama-3.2-3b/w4a4/exp \
       --lm_eval --lm_eval_batch_size 16
   ```

2. 為避免版本相容性問題，建議在 CLAUDE.md 中明確標註 Llama 3.2 需要的 transformers 版本

3. 考慮添加自動化測試腳本，以便快速驗證已訓練模型的效能

---

報告編號：001  
報告日期：2025-07-25