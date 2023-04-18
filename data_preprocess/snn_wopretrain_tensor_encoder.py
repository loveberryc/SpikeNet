import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate
import snntorch as snn
import numpy as np
import pickle
from torch.utils.data import DataLoader
from dataset import TensorDataset

class TextEmbedding(nn.Module):
    def __init__(self, word2id, hidden_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(word2id), hidden_dim)
    
    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        x = self.embedding(x)
        mean_value = torch.mean(x)
        variance_value = torch.var(x)
        x = torch.clip((x - mean_value) / 6 / torch.sqrt(variance_value) + 0.5, 0, 1)
        return x


def main(dataset_name, data_type, sentence_length):
    with open(f"../data/{dataset_name}/train_{dataset_name}.word2id", 'rb') as f:
        word2id = pickle.load(f)
    embedding_model = TextEmbedding(word2id, 300)

    saved_weights = torch.load(f"../ann_tailored_embed_models/{dataset_name}.pth")
    embedding_model.load_state_dict(saved_weights, strict=False)
    
    with open( f"../data/{dataset_name}/{data_type}_{dataset_name}_{sentence_length}_idx.tensor_dataset", 'rb') as f:
        dataset = pickle.load(f)
    
    embedding_model.to("cuda")
    embedding_tuple_list = []
    for data in dataset:
        x = embedding_model(data[0].to("cuda"))
        embedding_tuple_list.append((x.to("cpu"), data[1]))
    ret_tensor_dataset = TensorDataset(embedding_tuple_list)
    file_name = f"../data/{dataset_name}/{data_type}_{dataset_name}_{sentence_length}_embed.tensor_dataset"
    with open(file_name, 'wb') as f:
        pickle.dump(ret_tensor_dataset, f, -1)
    return

if __name__ == "__main__":
    main(
            dataset_name="sst2",
            sentence_length = 25,
            data_type = "train"
        )
    main(
            dataset_name="sst2",
            sentence_length = 25,
            data_type = "test"
        )
    main(
            dataset_name="mr",
            sentence_length = 35,
            data_type = "train"
        )
    main(
            dataset_name="mr",
            sentence_length = 35,
            data_type = "test"
        )
    main(
            dataset_name="subj",
            sentence_length = 35,
            data_type = "train"
        )
    main(
            dataset_name="subj",
            sentence_length = 35,
            data_type = "test"
        )
    main(
            dataset_name="sst5",
            sentence_length = 25,
            data_type = "train"
        )
    main(
            dataset_name="sst5",
            sentence_length = 25,
            data_type = "test"
        )
    main(
            dataset_name="senti",
            sentence_length = 32,
            data_type = "train"
        )
    main(
            dataset_name="senti",
            sentence_length = 32,
            data_type = "test"
        )
    main(
            dataset_name="waimai",
            sentence_length = 32,
            data_type = "train"
        )
    main(
            dataset_name="waimai",
            sentence_length = 32,
            data_type = "test"
        )