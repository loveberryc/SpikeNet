import sys
sys.path.append("..")
import re
import torch
import nltk
import os
import pickle
from tqdm import tqdm
import jieba
from dataset import TensorDataset

def main(dataset_name, language, max_len, data_type):
    # generate word2id vocab
    train_path = f"../data/{dataset_name}/train.txt"
    sentences = []
    word2id = {'pad': 0, 'unk': 1}
    with open(train_path, encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            sentence = re.sub(r"\-", " ", sentence)
            if language == "eng":
                sentence = nltk.word_tokenize(sentence)
            elif language == "chn":
                sentence = jieba.lcut(sentence)
            sentences.append(sentence)
    for sentence in sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
    file_name = f"../data/{dataset_name}/train_{dataset_name}.word2id"
    
    with open(file_name, 'wb') as f:
        pickle.dump(word2id, f, -1)

    samples = []
    file_path = f"../data/{dataset_name}/{data_type}.txt"
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('\t')
            sentence = temp[0].strip()
            label = int(temp[1])
            sentence = re.sub(r"\-", " ", sentence)
            if language == "eng":
                sentence = nltk.word_tokenize(sentence)
            elif language == "chn":
                sentence = jieba.lcut(sentence)
            samples.append((sentence, label))

    sentence2index_list = []
    for sample in samples:
        sentence = sample[0]
        label = sample[1]
        sentence_index = []
        for word in sentence:
            if word in word2id.keys():
                sentence_index.append(word2id.get(word))
            else: # unkown words
                sentence_index.append(word2id.get("unk"))
        if len(sentence_index) < max_len:
            sentence_index += (max_len - len(sentence_index)) * [0]
        else:
            sentence_index = sentence_index[:max_len]
        sentence2index_list.append((torch.tensor(sentence_index, dtype=int), label))
    dataset = TensorDataset(sentence2index_list)
    with open(f"../data/{dataset_name}/{data_type}_{dataset_name}_{max_len}_idx.tensor_dataset", 'wb') as f:
        pickle.dump(dataset, f, -1)
    
    

if __name__ == "__main__":
    main(
            dataset_name="sst2",
            language="eng",
            max_len = 25,
            data_type = "train"
        )
    main(
            dataset_name="sst2",
            language="eng",
            max_len = 25,
            data_type = "test"
        )
    main(
            dataset_name="mr",
            language="eng",
            max_len = 35,
            data_type = "train"
        )
    main(
            dataset_name="mr",
            language="eng",
            max_len = 35,
            data_type = "test"
        )
    main(
            dataset_name="subj",
            language="eng",
            max_len = 35,
            data_type = "train"
        )
    main(
            dataset_name="subj",
            language="eng",
            max_len = 35,
            data_type = "test"
        )
    main(
            dataset_name="sst5",
            language="eng",
            max_len = 25,
            data_type = "train"
        )
    main(
            dataset_name="sst5",
            language="eng",
            max_len = 25,
            data_type = "test"
        )
    main(
            dataset_name="senti",
            language="chn",
            max_len = 32,
            data_type = "train"
        )
    main(
            dataset_name="senti",
            language="chn",
            max_len = 32,
            data_type = "test"
        )
    main(
            dataset_name="waimai",
            language="chn",
            max_len = 32,
            data_type = "train"
        )
    main(
            dataset_name="waimai",
            language="chn",
            max_len = 32,
            data_type = "test"
        )