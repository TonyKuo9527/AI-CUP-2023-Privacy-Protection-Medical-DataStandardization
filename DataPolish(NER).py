import os


def read_file(path):
    with open(path , 'r' , encoding = 'utf-8-sig') as fr:
        return fr.readlines()


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
    file_name = txt_name + '.txt'
    sents = read_file(os.path.join(medical_report_folder, file_name))
    article = "".join(sents)


    bounary , item_idx , temp_seq, seq_pairs = 0 , 0 , '' , []
    new_line_idx = 0
    for w_idx, word in enumerate(article):

        if word == '\n':
            new_line_idx = w_idx + 1
            if article[bounary:new_line_idx] == '\n':
                continue
            if temp_seq == '':
                temp_seq = 'invalid[@]'
            sentence = article[bounary:new_line_idx].strip().replace('\t' , ' ')
            seq_pair = f"{sentence}\t{temp_seq}\n"
            bounary = new_line_idx
            if temp_seq != '':
                seq_pairs.append(seq_pair)
            temp_seq = ''
        if w_idx == annos_dict[txt_name][item_idx]['st_idx']:
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if 'normalize_time' in annos_dict[txt_name][item_idx]:
                temp_seq += phi_value + '[@]'
            else:
                temp_seq += phi_value + '[@]'
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

            if not(seq_pair.split('\t')[0]):
                continue

            if len(seq_pair.split('\t')[0]) > 512:
                continue

            fw.write(seq_pair)
    print("tsv format dataset done")

first_info_path = r"./First_Phase_Release/answer.txt"
first_folder = r"./First_Phase_Release/First_Phase_Text_Dataset"
tsv_output_path = './Data/train(NER).tsv'
generate_annotated_medical_report_parallel(first_info_path, first_folder, tsv_output_path, mode='w')
second_info_path = r"./Second_Phase_Release/answer.txt"
second_folder = r"./Second_Phase_Release/Second_Phase_Dataset"
tsv_output_path = './Data/train(NER).tsv'
generate_annotated_medical_report_parallel(second_info_path, second_folder, tsv_output_path, mode='a')
first_info_path = r"./Validation_Release/answer.txt"
first_folder = r"./Validation_Release/Validation_Text_Dataset"
tsv_output_path = './Data/valid(NER).tsv'
generate_annotated_medical_report_parallel(first_info_path, first_folder, tsv_output_path, mode='w')