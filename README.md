# AI CUP 2023 隱私保護與醫學數據標準化競賽: 解碼臨床病例、讓數據說故事v - Team_3170

## 運行環境
 - Python : 3.8.10
 - Pip : 20.0.2
 - PyTorch : 2.1.1+cu121
 - CUDA : 12.2
 - CUDA Toolkit : 12.2

### 運行裝置規格
 - OS : Ubuntu 20.04.6 LTS
 - CPU : I9 13900K
 - GPU : RTX4090 24G
 - Ram : 64G

### 套件版本
 - torch : 2.1.1+cu121
 - transformers : 4.35.2
 - tqdm : 4.65.0
 - datasets : 2.15.0

## 文本資料
- 由於原始文本資料數量眾多，採用額外下載方式取得，請另行下載並放入對應資料夾內導入(或是直接採用Data內我們提前處理好的數據)。

## 模型權重
|Model|Epoch|Input Max Length|Batch Size|Loss Function|Optimizer|Learning Rate|Direction|URL|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
|SpanBERT_13|13|512|11|default|AdamW|1e-05|實體識別|[Download](https://drive.google.com/file/d/1dakk9YvjQv-MgnHkmBp7ydjz13ekMApW/view?usp=sharing)|
|Robertalarge_10|10|512|11|default|AdamW|1e-05|文本分類|[Download](https://drive.google.com/file/d/1q2Wagt1DdFlWVEkRp0ETIZ280fGn1Ki-/view?usp=sharing)|
|T5large_12|12|512|1|default|AdamW|1e-05|時間數據正規化|[Download](https://drive.google.com/file/d/1g21pGekKsHj7uHIH41cAFJXuDl8y3RMx/view?usp=sharing)|

- 請將下載好的檔案放置`Models`資料夾，以確保程式能夠正常載入使用。

## 執行步驟(Train)
- 下載文本資料放置指定資料夾內(此步驟可略過，直接採用Data資料夾已處理完成的數據。若有下載請繼續往下執行)
- 執行DataPolish(NER).py (提取篩選實體識別用訓練及驗證資料)
- 執行Train(NER).py (訓練實體識別模型)
- 執行DataPolish(Multi-Class).py (提取篩選文本分類用訓練及驗證資料)
- 執行Train(Multi-Class).py (訓練文本分類模型)
- 執行DataPolish(Date).py (提取篩選時間數據正規化用訓練及驗證資料)
- 執行Train(Date).py (訓練時間數據正規化模型)


## 執行步驟(Test)
- 下載競賽資料和模型權重放置指定資料夾內 (若無下載，可直接使用opendid_test.tsv，此為事先處理並保存的資料，未下載需註解predict 307部分)
- 執行predict.py (進行競賽資料集預測)

## 聯絡信箱
G-Mail : tonykuo9527@gmail.com