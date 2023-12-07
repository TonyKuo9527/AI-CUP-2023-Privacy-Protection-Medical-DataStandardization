import os


def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()


def convert_type_to_number(type_name):
    type_to_number_mapping = {
        "PATIENT": 0, "DOCTOR": 1, "USERNAME": 2, "PROFESSION": 3,
        "ROOM": 4, "DEPARTMENT": 5, "HOSPITAL": 6, "ORGANIZATION": 7,
        "STREET": 8, "CITY": 9, "STATE": 10, "COUNTRY": 11, 
        "ZIP": 12, "LOCATION-OTHER": 13, "AGE": 14, "DATE": 15, 
        "TIME": 16, "DURATION": 17, "SET": 18, "PHONE": 19, 
        "FAX": 20, "EMAIL": 21, "URL": 22, "IPADDR": 23, "SSN": 24,
        "MEDICALRECORD": 25, "HEALTHPLAN": 26, "ACCOUNT": 27, 
        "LICENSE": 28, "VEHICLE": 29, "DEVICE": 30, "BIOID": 31, 
        "IDNUM": 32, "PHI":33
    }

    return type_to_number_mapping.get(type_name, None)
    

def process_annotation_file(lines):
    '''
    處理anwser.txt 標註檔案

    output:annotation dicitonary
    '''
    print("process annotation file...")
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif len(items) == 6:
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
                'normalize_time' : items[5],
            }

        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    print("annotation file done")
    return entity_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict):
    '''
    處理單個病理報告

    output : 處理完的 sequence pairs
    '''
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)


    bounary , item_idx , temp_seq , temp_type , temp, seq_pairs = 0 , 0 , [], [] , [], []
    new_line_idx = 0
    for w_idx, word in enumerate(article):

        if word == '\n':
            new_line_idx = w_idx + 1
            if article[bounary:new_line_idx] == '\n':
                continue
            if temp != []:
                for index, value in enumerate(temp_seq):
                    if (temp_type[index] in [15,16,17,19]):
                        seq_pair = f"{temp[index]}\t{value}\n"
                        seq_pairs.append(seq_pair)
            
            bounary = new_line_idx
            temp_seq = []
            temp = []
            temp_type = []
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp.append(phi_value)
                temp_seq.append(annos_dict[txt_name][item_idx]['normalize_time'])
                # temp_seq.append(phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time'])
                temp_type.append(convert_type_to_number(annos_dict[txt_name][item_idx]['phi']))
            if item_idx == len(annos_dict[txt_name]) - 1:
                continue
            item_idx += 1
    return seq_pairs


def contains_any_substring(string, substrings):
    for substring in substrings:
        if substring in string:
            return True
    return False

def generate_annotated_medical_report_parallel(anno_file_path, medical_report_folder , tsv_output_path, mode):
    '''
    呼叫上面的兩個function
    處理全部的病理報告和標記檔案

    output : 全部的 sequence pairs
    '''
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    txt_names = list(annos_dict.keys())

    all_seq_pairs = []

    for txt_name in txt_names:
        all_seq_pairs.extend(process_medical_report(txt_name, medical_report_folder, annos_dict))
    with open(tsv_output_path , mode , encoding = 'utf-8') as fw:
        for seq_pair in all_seq_pairs:
            seq_pair = (seq_pair.replace('\\n', ' '))

            if 'now' in (seq_pair.split('\t')[0]):
                continue

            if 'Now' in (seq_pair.split('\t')[0]):
                continue

            fw.write(seq_pair)
    print("tsv format dataset done")

first_info_path = r"./First_Phase_Release/answer.txt"
first_folder = r"./First_Phase_Release/First_Phase_Text_Dataset"
tsv_output_path = './Data/train(Date).tsv'
generate_annotated_medical_report_parallel(first_info_path, first_folder, tsv_output_path, mode='w')
second_info_path = r"./Second_Phase_Release/answer.txt"
second_folder = r"./Second_Phase_Release/Second_Phase_Dataset"
tsv_output_path = './Data/train(Date).tsv'
generate_annotated_medical_report_parallel(second_info_path, second_folder, tsv_output_path, mode='a')
first_info_path = r"./Validation_Release/answer.txt"
first_folder = r"./Validation_Release/Validation_Text_Dataset"
tsv_output_path = './Data/valid(Date).tsv'
generate_annotated_medical_report_parallel(first_info_path, first_folder, tsv_output_path, mode='w')