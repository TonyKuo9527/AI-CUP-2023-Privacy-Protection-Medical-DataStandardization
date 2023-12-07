import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import random
from datasets import load_dataset, Features, Value, concatenate_datasets

# 設定
plm = 'roberta-large'
plm_name = 'Robertalarge'
learning_rate = 1e-5
train_data = './Data/train(Multi-Class).tsv'
valid_data = './Data/valid(Multi-Class).tsv'
batch_size = 11
model_dir = f"./Models"
max_length = 512


tokenizer = RobertaTokenizer.from_pretrained(plm)
model = RobertaForSequenceClassification.from_pretrained(plm, num_labels=21, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)


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
    inputs = [e + '[SEP]' + c for e, c in zip(examples["extract"], examples["content"])]
    labels = examples["label"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    # 設置目標
    labels = [int(label) for label in labels]
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
                        features=Features({'content': Value('string'), 'extract': Value('string'), 'label': Value('int64')}),
                        column_names=['content', 'extract', 'label'], keep_default_na=False)
valid_dataset = load_dataset("csv", data_files=valid_data, delimiter='\t',
                        features=Features({'content': Value('string'), 'extract': Value('string'), 'label': Value('int64')}),
                        column_names=['content', 'extract', 'label'], keep_default_na=False)


combined_dataset = concatenate_datasets([train_dataset['train'], valid_dataset['train']])
# 打亂數據集
combined_dataset = combined_dataset.shuffle(seed=15)
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
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
        for i, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            total_val_correct += correct
            total_val_examples += labels.size(0)


    avg_val_loss = total_val_loss / len(valid_dataloader)
    val_accuracy = total_val_correct / total_val_examples
    print(f'Epoch {epoch+1}, Validation loss: {avg_val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')
    
    scheduler.step()

    if  val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        model_path = os.path.join(model_dir, f'{plm_name}_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)