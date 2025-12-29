import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

import time as mytimer



import torch.nn.functional as F

from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
from torch.distributions.uniform import Uniform # So we can initialize our tensors with a uniform distribution

import pandas as pd ## to create dataframes from graph input
import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.

import spacy as spacy
import string
from torch.nn.utils.rnn import pad_sequence
import json
import os
import random
torch.autograd.set_detect_anomaly(True)
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

### our self writed code:
import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network and allows
                      # us to access a lot of useful functions like:
                      # nn.Linear, nn.Embedding, nn.CrossEntropyLoss() etc.

from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
from torch.distributions.uniform import Uniform # So we can initialize our tensors with a uniform distribution
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L # lightning has tons of cool tools that make neural networks easier

import pandas as pd ## to create dataframes from graph input
import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.
#import cupy as cp
import re
import string
import numpy as np
import torch.nn.functional as F
# from scipy.special import softmax
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import timeit
from math import ceil
import json

from time import time

import sys

BATCH_SIZE = 1
EMBEDD_DIM = 250
ENCODING_SCALE = 6
WORD_PREDICT_NUMBER = 1
EPOCH_NUMBER = 5
PORT_NUMBER = 48600
LEARNING_RATE = 0.008
LEARNING_RATE_FACTOR = 0.994
PROCESS_NUMBER = 4

def remove_punctuation(text):
        exclude = set(string.punctuation + "\t") - {"'"}
        translator = str.maketrans("", "", "".join(exclude))
        cleaned_text = text.translate(translator)
        return cleaned_text

def find_vocabs(only_vocab = 0):
    global vocab_size
    global tokens_size

    vocab = {
        }
    if(os.path.exists('models/inputs.pth')):
        inputs = torch.load('models/inputs.pth')
        labels = torch.load('models/labels.pth')
        with open('models/vocab.json', 'r') as file:
            vocab = json.load(file)
        vocab_size = len(vocab)
        with open('models/tokens_size.txt', 'r') as file:
            for line in file:
                tokens_size = int(line)
    else:
    
        new_inputs = []
        new_labels = []
        
        tokens_size = 0

        file_path = "dataset.txt"
        directory_path = 'new_dataset'
        old_vocab_size = len(vocab)
        print(old_vocab_size)
        # Loop through each file in the directory and read it line by line
        for entry in os.scandir(directory_path):
            if entry.is_file():
                print(f"Reading lines from {entry.name}:")
                
                with open(entry.path, 'r') as file:
                    totalLine = 0
                    for line in file:
                        if line.strip() == '':
                            continue
                        text = remove_punctuation(line)
                        text = text.replace("\n", "")
                        text = text.lower()

                        nlp = spacy.load('en_core_web_sm')

                        # Tokenize the string using spaCy
                        tokens = [token.text for token in nlp(text)]
                        tokens_size += len(tokens)
                
                        for  index, item in enumerate(tokens):
                            if item not in vocab:
                                # Assign a new index to the token
                                vocab[item] = len(vocab)
                        if '<EOS>' not in vocab:
                            vocab['<EOS>'] = len(vocab)
                        totalLine += 1    
                        print(totalLine)

        
        vocab_size = len(vocab)
        # print("vocab", vocab)
        if only_vocab == 1:
            return vocab

        # print("input length: ", tokens_size)
        # print("vocab size: ", vocab_size)

        for entry in os.scandir(directory_path):
            if entry.is_file():
                print(f"Reading lines from {entry.name}:")

                with open(entry.path, 'r') as file:
                    for line in file:
                        if line.strip() == '':
                            continue
                        text = remove_punctuation(line)
                        text = text.replace("\n", "")
                        text = text.lower()

                        nlp = spacy.load('en_core_web_sm')

                        # Tokenize the string using spaCy
                        tokens = [token.text for token in nlp(text)]

                        input_list = []
                        label_list = []

                        for i in range(len(tokens)-1):
                            
                                input_list.append(F.one_hot(torch.tensor(vocab[tokens[i]]), vocab_size).to(dtype=torch.float))
                                label_list.append(F.one_hot(torch.tensor(vocab[tokens[i+1]]), vocab_size).to(dtype=torch.float))

                        input_list.append(F.one_hot(torch.tensor(vocab[tokens[len(tokens)-1]]), vocab_size).to(dtype=torch.float))
                        label_list.append(F.one_hot(torch.tensor(vocab['<EOS>']), len(vocab)).to(dtype=torch.float))
                        # print(input_list)
                        input_stack = torch.stack(input_list)
                        label_stack = torch.stack(label_list)
                        new_inputs.append(input_stack)
                        new_labels.append(label_stack)

        inputs = pad_sequence(new_inputs, batch_first=True, padding_value=0)
        labels = pad_sequence(new_labels, batch_first=True, padding_value=0)

        torch.save(inputs, 'models/inputs.pth')
        torch.save(labels, 'models/labels.pth')
        outf = open('models/tokens_size.txt', 'w')
        outf.write(str(tokens_size))
        outf.close()
        with open('models/vocab.json', 'w') as file:
            json.dump(vocab, file)

    print(len(vocab))
    # print("inputs: ", inputs)
    # print("labels:", labels)

    return inputs, labels, vocab

    # dataset = TensorDataset(inputs, labels)
    # dataloader = DataLoader(dataset)

class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self):
        super().__init__()
        

        L.seed_everything(seed=66)
        
        self.embedding_dim = 256
   
        self.input_to_wordembeding = nn.Linear(in_features=vocab_size, out_features=self.embedding_dim, bias=False)
        self.wordembeding_to_query = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)
        self.wordembeding_to_key = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)
        self.wordembeding_to_value = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False)     
        self.selfattention_to_finalweights = nn.Linear(in_features=self.embedding_dim, out_features=vocab_size, bias=True)

        max_sequence_length = tokens_size
        
        self.positional_encoding = self.positional_encoding(max_sequence_length, self.embedding_dim)
        

        self.query_list = []
        self.key_list = []
        self.value_list = []
        self.iteration_weights = []
        self.similarity = []
        self.result = []
        self.position_index = 0

    def forward(self, input):
        self.query_list = []
        self.key_list = []
        self.value_list = []
        self.iteration_weights = []
        self.similarity = []
        self.result = []
        # print("input: ", input)
        # print("query gradients: ", self.wordembeding_to_query.weight.grad)
        # print("key gradients: ", self.wordembeding_to_key.weight.grad)
        # print("value gradients: ", self.wordembeding_to_value.weight.grad)
        data = {
            "q1": self.wordembeding_to_query.weight.detach()[0].numpy(), # [0] = Weights to top activation function
            "q2": self.wordembeding_to_query.weight.detach()[1].numpy(), # [1] = Weights to bottom activation function
            "k1": self.wordembeding_to_key.weight.detach()[0].numpy(), # [0] = Weights to top activation function
            "k2": self.wordembeding_to_key.weight.detach()[1].numpy(), # [1] = Weights to bottom activation function
            "v1": self.wordembeding_to_value.weight.detach()[0].numpy(), # [1] = Weights to bottom activation function
            "v2": self.wordembeding_to_value.weight.detach()[1].numpy(), # [1] = Weights to bottom activation function
            # "token": ["what", "is", "statquest", "<EOS>", "awesome", "how", "dog", "nice"],
            # "input": ["input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8"]
        }
        df = pd.DataFrame(data)
        # print(df)

        embedded_word_list = []
        positional_embedded_word_list = []
        weights_list = []
        attention_list = []
        residualConnection_list = []
        output_values_list = []

        valid_one_hot_list = list(filter(self.is_valid_one_hot, input[0]))
        
        for idx, item in enumerate(valid_one_hot_list):
            # print("item: ", item)
            
            embedded_word = self.input_to_wordembeding(item)
            embedded_word = F.relu(embedded_word)
            embedded_word_list.append(embedded_word)
            positional_embedded_word = embedded_word_list[idx] + self.positional_encoding[0][idx]
            positional_embedded_word_list.append(positional_embedded_word)

            q = self.wordembeding_to_query(positional_embedded_word_list[idx])
            k = self.wordembeding_to_key(positional_embedded_word_list[idx])
            v = self.wordembeding_to_value(positional_embedded_word_list[idx])

            self.query_list.append(q)
            self.key_list.append(k)
            self.value_list.append(v)

            # print("query list: ", self.query_list)
            # print("key list: ", self.key_list)
            # print("value list: ", self.value_list)
            result_list = [torch.matmul(q, key) for key in self.key_list]
            result_stack = torch.stack(result_list)

        
            weights = torch.softmax(result_stack.view(1,-1), dim=1)[0]
            weights_list.append(weights)

            
            # print("mutliply q*k result: ", result_stack)
            # print("weights: ", weights)
            # print("weights list: ", weights_list)
            
            attention = torch.zeros_like(positional_embedded_word, requires_grad=True)
            for i in range(idx + 1):
                attention = attention + weights_list[idx][i] * self.value_list[i]
            attention_list.append(attention)
            residualConnection = attention_list[idx] + positional_embedded_word_list[idx]
            # print("attention: ", attention)
            # print("positional_embedded: ", positional_embedded_word_list[idx])
            # print("residualConnection: ", residualConnection)
            residualConnection_list.append(residualConnection)
            output_values = self.selfattention_to_finalweights(residualConnection_list[idx])
            output_values_list.append(output_values)

        # print("input Start------------------------------------------------------------------------------------------------------------")
        # print("embedded word:", embedded_word_list)
        # print("positional:", positional_embedded_word_list)
        # print("query:", self.query_list)
        # print("key:", self.key_list)
        # print("value:", self.value_list)
        # print("weights:", weights_list)
        # print("attention:", attention_list)
        # print("residual:", residualConnection_list)
        # print("outputs:", output_values_list)
        # print("input End--------------------------------------------------------------------------------------------------------------")

        return torch.stack(output_values_list)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # print("output: ", y_hat)
        # print("label: ", y)
        # result = torch.softmax(y_hat, dim=1)
        valid_one_hot_list = list(filter(self.is_valid_one_hot, y[0]))
        # print("result: ", result)
        # loss = nn.MSELoss()(result, torch.stack(valid_one_hot_list).float())
        loss = nn.CrossEntropyLoss()(y_hat, torch.stack(valid_one_hot_list).float())
        # criterion = nn.BCEWithLogitsLoss()
        # loss = criterion(result, torch.stack(valid_one_hot_list).float())
        loss.backward(retain_graph=True)
        # self.clear_intermediate_values()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        print("loss: ", loss, f'Learning Rate: {current_lr}')
        return loss

    def configure_optimizers(self):
        with open('models/lr.txt', 'r') as file:
            for line in file:
                lr = float(line)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: LEARNING_RATE_FACTOR ** epoch)
        # self.lr_scheduler = lr_scheduler.LinearLR(self.optimizer,start_factor=1,end_factor=0,total_iters=EPOCH_NUMBER)
        return [self.optimizer], [self.lr_scheduler]
    
    def on_train_epoch_end(self):
        average_gradients(self)

    def positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Calculate the sine and cosine components
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc.unsqueeze(0)
    
    def clear_intermediate_values(self):
    # Clear intermediate values to release memory
        del self.query_list[:]
        del self.key_list[:]
        del self.value_list[:]
        del self.iteration_weights[:]
        del self.similarity[:]
        del self.result[:]
    
    def is_valid_one_hot(self, encoded_vector):
        # Check if the vector is binary and has exactly one 1
        return all(bit in [0, 1] for bit in encoded_vector) and encoded_vector.tolist().count(1) == 1

def predict_sentences(text,model):
    text = remove_punctuation(text)
    text = text.lower()
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')

    # Tokenize the string using spaCy
    tokens = [token.text for token in nlp(text)]

    print("Original String:", text)
    print("Tokens:", tokens)

    vocab = find_vocabs(only_vocab=1)

    prompt_list = []
    # print("vocab", vocab)
    # print("prompt list:", prompt_list)
    for i in range(len(tokens)):
        # print("token:", tokens[i], "token number:" , vocab[tokens[i]])
        token = F.one_hot(torch.tensor(vocab[tokens[i]]), len(vocab)).to(dtype=torch.float)
        # print("one hot", token)
        prompt_list.append(token)
        # print(tokens[i], end=" ")

    # print(prompt_list)
    prompt_stack = torch.stack(prompt_list)
    prompt = torch.unsqueeze(prompt_stack, dim=0)

    # print(prompt)


    softmax = nn.Softmax(dim=0) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns
    reverse_vocab = {index: word for word, index in vocab.items()}

    print("rank",dist.get_rank(),":")
    for i in range(5):
        result = model(prompt)
        # print(result)
        indices = torch.argmax(result, dim=1)
        prompt_list.append(F.one_hot(torch.tensor(indices[len(indices)-1]), len(vocab)).to(dtype=torch.float))
        prompt_stack = torch.stack(prompt_list)
        prompt = torch.unsqueeze(prompt_stack, dim=0)
        print(reverse_vocab[indices[len(indices)-1].item()], end= " ")
    print()



class Partition(object): 

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):

        data_idx = self.index[index]
        return (self.data[0][data_idx],self.data[1][data_idx])


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.5, 0.5], seed=1234):
        self.data = data # data = ([40,180,180],[40,180,180]) 
        self.partitions = []
        data_len = len(data[0])
        indexes = [x for x in range(0, data_len)]

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
  

    def use(self, partition):
        # print("self.partitions:", self.partitions)
        return Partition(self.data, self.partitions[partition])

""" Partitioning MNIST """
def partition_dataset():
    inputs, labels, vocabs = find_vocabs()
    size = dist.get_world_size()
    dataset = (inputs, labels)
    bsz = BATCH_SIZE
    partition_sizes = [1.0 / size for _ in range(size)] 
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=False)
    return train_set, bsz, vocabs

def temp_run(rank, size):
    torch.manual_seed(1234)
    print(dist.get_rank())
    train_set, bsz, vocab = partition_dataset()
    # print(len(train_set))
    model = DecoderOnlyTransformer()
    if(os.path.exists('models/model.pt')):
        model.load_state_dict(torch.load('models/model.pt'))
    optimizer, lr_sceduler = model.configure_optimizers()
    optimizer = optimizer[0]
    lr_sceduler = lr_sceduler[0]
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    start_time = time()
    for epoch in range(EPOCH_NUMBER):
        epoch_loss = 0.0
        print("epoch ", epoch, "started ...")
        for (data, target) in train_set:
            optimizer.zero_grad()
            loss = model.training_step((data, target),0)
            epoch_loss += loss.item()
            average_gradients(model)
            optimizer.step()
        lr_sceduler.step()
        if(dist.get_rank() % PROCESS_NUMBER == 0):
            torch.save(model.state_dict(),"models/model.pt")
            with open("models/vocab.txt","w") as fp:
                vocab['token_size'] = tokens_size
                json.dump(vocab,fp)
        mytimer.sleep(45)
    end_time = time()
    print('Rank ', dist.get_rank(), ', epoch ',
        epoch, ': ', epoch_loss / num_batches, ', execution time: ', end_time - start_time)    
    


    
    """ Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    group = dist.new_group([0,1,2,3,4,5,6,7,8,9,10,11])
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM,group=group)
        param.grad.data /= size



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.39.18'
    os.environ['MASTER_PORT'] = str(PORT_NUMBER)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    args = sys.argv[1:]
    rank = int(args[0])
    size = 12
    mp.set_start_method("spawn")

    while(True):
        # print("learning rate: ", LEARNING_RATE)
        
        p = mp.Process(target=init_process, args=(rank, size, temp_run))
        p.start()
        p.join()
        print("process joined!")
        PORT_NUMBER += 1 
        LEARNING_RATE = LEARNING_RATE * pow(LEARNING_RATE_FACTOR, EPOCH_NUMBER)
        outf = open('models/lr.txt', 'w')
        outf.write(str(LEARNING_RATE))
        outf.close()