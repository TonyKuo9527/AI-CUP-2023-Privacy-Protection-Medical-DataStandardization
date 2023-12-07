import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import random
from datasets import load_dataset, Features, Value, concatenate_datasets

# 設定
plm = 'SpanBERT/spanbert-large-cased'
plm_name = 'SpanBERT'
learning_rate = 1e-5
train_data = './Data/train(NER).tsv'
valid_data = './Data/valid(NER).tsv'
batch_size = 11
model_dir = f"./Models"
max_length = 512


tokenizer = AutoTokenizer.from_pretrained(plm)
model = AutoModelForTokenClassification.from_pretrained(plm, num_labels=3, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)

# 固定隨機種子
def set_torch_seed(seed=15):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_torch_seed()


# 根據數據集需要調整的數據預處理函數
def preprocess_function(examples):
    inputs = examples["content"]
    extracts = examples["extract"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')

    labels = []
    for content, extracts in zip(inputs, extracts):
        extracts = extracts.split("[@]")[:-1]
        tokenized_extracts = []
        extracts_length = []

        for extract in extracts:
            tokenized_extract = tokenizer(extract, max_length=max_length, truncation=True, padding='max_length')['input_ids']
            tokenized_extract = [token for token in tokenized_extract if token not in [101, 102, 0]]
            extracts_length.append(len(tokenized_extract))
            tokenized_extracts.append(tokenized_extract)

        tokenized_content = tokenizer(content, max_length=max_length, truncation=True, padding='max_length')['input_ids']

        label = [0] * len(tokenized_content)

        for i in range(len(tokenized_content)):
            if (tokenized_content[i] == 0 or
                tokenized_content[i] == 101 or
                tokenized_content[i] == 102):
                continue

            for index, extract_length in enumerate(extracts_length):
                if tokenized_content[i:i + extract_length] == tokenized_extracts[index]:
                    label[i] = 1
                    for j in range(1, extract_length):
                        label[i + j] = 2
                    break
        labels.append(label)


    model_inputs["labels"] = labels
    return model_inputs


def collate_fn(batch):
    # 將列表的元素（字典）合併為單一個字典
    batch_input_ids = [item['input_ids'] for item in batch]
    batch_attention_mask = [item['attention_mask'] for item in batch]
    batch_labels = [item['labels'] for item in batch]

    # 轉換為張量
    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }


# 載入數據集
train_dataset = load_dataset("csv", data_files=train_data, delimiter='\t',
                        features=Features({'content': Value('string'), 'extract': Value('string')}),
                        column_names=['content', 'extract'], keep_default_na=False)
valid_dataset = load_dataset("csv", data_files=valid_data, delimiter='\t',
                        features=Features({'content': Value('string'), 'extract': Value('string')}),
                        column_names=['content', 'extract'], keep_default_na=False)


combined_dataset = concatenate_datasets([train_dataset['train'], valid_dataset['train']])
# 打亂數據集
combined_dataset = combined_dataset.shuffle(seed=42)
split_datasets = combined_dataset.train_test_split(test_size=0.1)

train_tokenized_dataset = split_datasets['train'].map(preprocess_function, batched=True)
valid_tokenized_dataset = split_datasets['test'].map(preprocess_function, batched=True)


# 設置訓練設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_dataloader = DataLoader(train_tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 訓練配置
epochs = 15 
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)


no_improve_epochs = 0  # 用於跟蹤沒有改善的epoch數
best_accuracy = 0


# 訓練循環
model.train()
for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    total_examples = 0
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)


        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 計算正確率
        preds = outputs.logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_examples += labels.size(0)

        loss.backward()
        optimizer.step()


    model.eval()  # 切換模型到評估模式
    total_val_loss = 0
    total_val_correct = 0
    total_val_examples = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            
            # 更新正確率的計算方法
            batch_correct = torch.all(preds == labels, dim=1).sum().item()  # 檢查每個序列是否完全匹配
            total_val_correct += batch_correct
            total_val_examples += input_ids.size(0)  # 總序列數


    avg_val_loss = total_val_loss / len(valid_dataloader)
    val_accuracy = total_val_correct / total_val_examples
    print(f'Epoch {epoch+1}, Validation loss: {avg_val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')
    
    scheduler.step()

    if  val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        model_path = os.path.join(model_dir, f'{plm_name}_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)