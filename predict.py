import os
import torch
import re
import datetime

from tqdm import tqdm
from dateutil import parser

from transformers import (
    RobertaForSequenceClassification, AutoTokenizer,
    T5ForConditionalGeneration, T5Tokenizer, AutoModelForTokenClassification,
    RobertaTokenizerFast
)


#1
model_1_path = './Models/SpanBERT_13.pt'
model_1 = AutoModelForTokenClassification.from_pretrained('SpanBERT/spanbert-large-cased', num_labels=3)
tokenizer_1 = AutoTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
model_1.load_state_dict(torch.load(model_1_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1.to(device)
model_1.eval()


#2
model_2_path = './Models/Robertalarge_10.pt'
model_2 = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=21)
tokenizer_2 = RobertaTokenizerFast.from_pretrained('roberta-large')
model_2.load_state_dict(torch.load(model_2_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_2.to(device)
model_2.eval()


#3(date)
model_3_path = './Models/T5large_12.pt'
model_3 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
tokenizer_3 = T5Tokenizer.from_pretrained('google/flan-t5-large')
model_3.load_state_dict(torch.load(model_3_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_3.to(device)
model_3.eval()



def predict1(text, max_length=512):
    # 對文本進行編碼，包括偏移映射
    inputs = tokenizer_1.encode_plus(text, max_length=max_length, truncation=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    offset_mapping = inputs["offset_mapping"][0]  # 獲取偏移映射

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 使用模型進行預測
    with torch.no_grad():
        outputs = model_1(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 將logits轉換為具體的預測類別
    predictions = torch.argmax(logits, dim=-1)[0]

    # 提取多個內容
    extracted_info = []  # 用於存儲提取內容及其索引的字典列表
    start, end = None, None
    for idx, pred in enumerate(predictions.cpu().numpy()):
        if pred == 1:
            if start is not None and end is not None:
                # 保存之前的提取內容及其索引
                extracted_info.append({'extract': text[start:end], 'start': start, 'end': end})
            # 設置新的開始標記
            start = offset_mapping[idx][0]
            end = None
        elif start is not None and pred != 2:
            # 設置結束標記
            end = offset_mapping[idx-1][1]
            # 保存當前的提取內容及其索引
            extracted_info.append({'extract': text[start:end], 'start': start, 'end': end})
            start, end = None, None
        elif start is not None and pred == 2 and idx == len(predictions) - 1:
            # 處理結束於標記2的情況
            end = offset_mapping[idx][1]
            extracted_info.append({'extract': text[start:end], 'start': start, 'end': end})

    return extracted_info


def predict2(data, max_length=512):
    text = data["extract"] + '[SEP]' + data["content"]
    inputs = tokenizer_2.encode_plus(text, max_length=max_length, truncation=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model_2(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)[0]

    return predictions.cpu().numpy().item()


def predict3(text, max_length=512):
    inputs = tokenizer_3.encode_plus(text, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 使用模型进行预测
    model_3.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        outputs = model_3.generate(input_ids=input_ids, attention_mask=attention_mask)

    # 解码生成的文本
    decoded_output = tokenizer_3.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


def convert_number_to_type(number):
    number_to_type_mapping = {
        0: "PATIENT", 1: "DOCTOR", 2: "ROOM", 3: "DEPARTMENT",
        4: "HOSPITAL", 5: "ORGANIZATION", 6: "STREET", 7: "CITY", 
        8: "STATE", 9: "COUNTRY", 10: "ZIP", 11: "LOCATION-OTHER", 
        12: "AGE", 13: "DATE", 14: "TIME", 15: "DURATION", 
        16: "SET", 17: "PHONE", 18: "URL", 19: "MEDICALRECORD", 
        20: "IDNUM"
    }

    return number_to_type_mapping.get(number, None)


def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()


def process_data(test_txts , out_file):
    with open(out_file , 'w' , encoding = 'utf-8') as fw:
        for txt in test_txts:
            m_report = read_file(txt)
            boundary = 0
            fid = txt.split('/')[-1].replace('.txt' , '')
            for idx,sent in enumerate(m_report):
                if sent.replace(' ' , '').replace('\n' , '').replace('\t' , '') != '':
                    sent = sent.replace('\t' , ' ')
                    fw.write(f"{fid}\t{boundary}\t{sent}\n")
                boundary += len(sent)


def find_keyword_indices(text, keyword):
    keyword_start_index = text.find(keyword)

    if keyword_start_index != -1:
        keyword_end_index = keyword_start_index + len(keyword)
        return keyword_start_index, keyword_end_index
    else:
        return None, None


def read_tsv(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                if len(fields) == 3:
                    data.append(fields)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def format_date_time(date_string):
    # Convert 12-hour format to 24-hour format
    date_string = date_string.replace('at', '').strip()
    date_string = date_string.replace('hrs', '').strip()
    date_string = date_string.replace('Hrs', '').strip()
    date_string = date_string.replace('hr', '').strip()
    date_string = date_string.replace('the', '').strip()
    date_string = date_string.replace('art', '').strip()
    date_string = date_string.replace(' o ', 'on').strip()
    date_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_string)


    date_string = re.sub(
        r'(\d{1,2}):(\d{2})pm',
        lambda m: f"{int(m.group(1)) + 12 if int(m.group(1)) < 12 else m.group(1)}:{m.group(2)}",
        date_string,
        flags=re.IGNORECASE
    )
    date_string = re.sub(r'(\d{1,2}):(\d{2})am',
                        lambda m: f"{m.group(1)}:{m.group(2)}" if m.group(1) != '12' else f"00:{m.group(2)}",
                        date_string,
                        flags=re.IGNORECASE)

    date_string = re.sub(
        r'(\d{2})\.(\d{2}) on (\d{1,2})/(\d{1,2})/(\d{2})',
        lambda m: f"{m.group(3)}/{m.group(4)}/20{m.group(5)} {m.group(1)}:{m.group(2)}",
        date_string
    )

    date_string = re.sub(
        r'(\d{4})(\d{2})(\d{2})',
        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
        date_string
    )

    # Normalize different date and time formats
    date_string = re.sub(r'(\d+)\.(\d+)(am|pm|on)', r'\1:\2\3', date_string)
    date_string = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1/\2/\3', date_string)
    if not(':' in date_string):
        date_string = re.sub(r'(\d{2})(\d{2})\s', r'\1:\2 ', date_string)
        date_string = re.sub(r'(\d{2})(\d{2})(am|pm|on)', r'\1:\2 ', date_string)

    return date_string

def is_specific_format(date_string, pattern):
    return bool(re.match(pattern, date_string.strip()))

def convert_to_iso_duration(text):

    match = re.match(r'(\d+)\s*(years?|yr?)', text, re.IGNORECASE)
    if match:
        years = match.group(1)
        return f"P{years}Y"
    
    match = re.match(r'(\d+)\s*(month?|months?)', text, re.IGNORECASE)
    if match:
        months = match.group(1)
        return f"P{months}M"
    
    match = re.match(r'(\d+)-(\d+)\s*(month?|months?)', text, re.IGNORECASE)
    if match:
        start_month = match.group(1)
        end_month = match.group(2)
        return f"P{start_month}.{end_month}M"
    
    match = re.match(r'(\d+)\s*weeks?', text, re.IGNORECASE)
    if match:
        number = match.group(1)
        return f"P{number}W"
    


    return None

def is_complete_date_format(date_string):
    if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$', date_string):
        return True
    
    if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$', date_string):
        return True
    
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_string):
        return True
    return False

def parse_generic_date(date_string):
    formatted_string  = format_date_time(date_string)
    if "00:00:00" in formatted_string :
        try:
            parsed_date = parser.parse(formatted_string)
            return parsed_date.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None



    if is_specific_format(formatted_string , r'^\d+(\s*-\s*\d+)?\s*(month?|months?|years?|yr?|weeks?)$'):
        return convert_to_iso_duration(formatted_string)

    if is_specific_format(formatted_string , r'^\d{4}$'):
        return formatted_string 

    if is_specific_format(formatted_string , r'^[a-zA-Z]+\s+\d{4}$'):
        try:
            parsed_date = parser.parse(formatted_string)
            return parsed_date.strftime("%Y-%m")
        except ValueError:
            return None

    formatted_string = format_date_time(formatted_string)

    if is_complete_date_format(formatted_string):
        return formatted_string
    else:
        try:
            parsed_date = parser.parse(formatted_string, dayfirst=True)
            return parsed_date.strftime("%Y-%m-%dT%H:%M") if parsed_date.time() != datetime.time(0, 0) else parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            return None

#未下載競賽資料請註解這段
test_phase_path = r'./opendid_test/opendid_test/'
out_file_path = './opendid_test.tsv'
test_txts = list(map(lambda x:os.path.join(test_phase_path , x) , os.listdir(test_phase_path)))
test_txts = sorted(test_txts)
process_data(test_txts , out_file_path)
#未下載競賽資料請註解這段

test_file_path = './opendid_test.tsv'
test_list = read_tsv(test_file_path)

with open("./answer.txt",'w',encoding='utf8') as fw:
    outputs = []
    for i, valid in tqdm(enumerate(test_list), total=len(test_list)):
        fid = valid[0]
        idx = valid[1]
        content = valid[2]

        result_list_1 = predict1(content)

        if result_list_1 != []:
            for result_1 in result_list_1:
                result_2 = predict2({
                    'extract' : result_1['extract'],
                    'content' : content
                })
                result_type = convert_number_to_type(result_2)
                start = result_1['start'] + int(idx)
                end = result_1['end'] + int(idx)
                if result_2 in [13,14,15,16]:
                    normalize_time = parse_generic_date(result_1['extract'])
                    if normalize_time == None:
                        normalize_time = predict3(result_1['extract'])
                    if normalize_time:
                        fw.write(f"{fid}\t{result_type}\t{start}\t{end}\t{result_1['extract']}\t{normalize_time}")
                        fw.write('\n')
                else:
                    fw.write(f"{fid}\t{result_type}\t{start}\t{end}\t{result_1['extract']}")
                    fw.write('\n')