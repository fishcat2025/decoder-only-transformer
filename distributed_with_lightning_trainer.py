""" Dataset partitioning helper """
from math import ceil
from random import random
import string

import os
from typing import Any, Dict


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import lightning as L # lightning has tons of cool tools that make neural networks easier
from lightning.pytorch.callbacks import ModelCheckpoint






import spacy as spacy
import sys



vocab = {}
def remove_punctuation(text):
    exclude = set(string.punctuation + "\t") - {"'"}
    translator = str.maketrans("", "", "".join(exclude))
    cleaned_text = text.translate(translator)
    return cleaned_text

def remove_punctuation(text):
        exclude = set(string.punctuation + "\t") - {"'"}
        translator = str.maketrans("", "", "".join(exclude))
        cleaned_text = text.translate(translator)
        return cleaned_text

def find_vocabs(only_vocab = 0):

    vocab = {
        }
    # with open('vocab.json', 'r') as file:
    #     vocab = json.load(file)
    print(len(vocab))

   

    new_inputs = []
    new_labels = []
    global tokens_size
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

    global vocab_size
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

    # print("inputs: ", inputs)
    # print("labels:", labels)

    return inputs, labels, vocab

def generate_dataset(inputs, labels):
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)
    return dataset, dataloader
    # dataset = TensorDataset(inputs, labels)
    # dataloader = DataLoader(dataset)




class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.automatic_optimization = False
        

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
        # df = pd.DataFrame(data)
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
        opt = self.optimizers()
        opt.zero_grad()
        x, y = batch
        y_hat = self.forward(x)
        # self.optimizer.zero_grad()
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
        # current_lr = self.optimizer.param_groups[0]['lr']
        print("loss: ", loss)
        # print("loss: ", loss, f'Learning Rate: {current_lr}')
        average_gradients(self)
        opt.step()

        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
        return self.optimizer
    
    
    def positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Calculate the sine and cosine components
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc.unsqueeze(0)
    def on_train_epoch_end(self):
        pass
        # print(self.training_step_outputs)
        # print(self.training_step_outputs.size)
        # avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()  # Calculate the average loss
        # print("avg-train-loss", avg_loss) # log training average loss
        # self.scheduler.step(avg_loss)  # Update the scheduler
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass
        # return super().on_save_checkpoint(checkpoint)
    
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

    _, _,vocab = find_vocabs(only_vocab=1)

    prompt_list = []
    print("vocab", vocab)
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
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, world_size, seed=1234):
        self.data = data
        self.partitions = []
        data_size = float(1.0 / world_size)
        sizes = []
        for _ in range(world_size):
            sizes.append(data_size)
        # print(sizes)    
        # rng = random()
        # rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        # rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        # print("my partitions are: ", self.partitions)
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset():
    # dataset = datasets.MNIST('./data', train=True, download=True,
    #                          transform=transforms.Compose([
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.1307,), (0.3081,))
    #                          ]))
    size = dist.get_world_size()
    bsz = 1
    global vocab
    inputs, labels, vocab = find_vocabs()
    dataset, dataLoader = generate_dataset(inputs, labels)
    # partition_sizes = [1.0 / size for _ in range(size)]
    # print(partition_sizes)
    rank = dist.get_rank()
    partition = DataPartitioner(dataset, size)
    # data = data
    # partitions = []
    # data_size = float(1.0 / size)
    # sizes = []
    # for _ in range(size):
    #     sizes.append(data_size)
    # for frac in sizes:
    #         part_len = int(frac * len(inputs))
    #         partitions.append(indexes[0:part_len])
    #         indexes = indexes[part_len:]
    partition = partition.use(rank)
    print("rank", rank)
    print("partion : ", partition)
    train_set = DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=False)
    
    # print("train_set", train_set)
    
    return train_set, bsz

def average_gradients(model):
    """Average gradients across all processes."""
    print("rank", dist.get_rank())
    for param in model.parameters() :
        if param.requires_grad and param is not None:
            # print("average runned")
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= float(dist.get_world_size())
# def average_gradients(model):
    # size = float(dist.get_world_size())
    # for param in model.parameters():
    #     dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #     param.grad.data /= size

def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = DecoderOnlyTransformer()
    # model.__init__()
    callbacks = [] # callbacks?
    mc = ModelCheckpoint(
        every_n_epochs=1, # monitors are checked every every_n_epochs epochs
        save_top_k=-1,  # save model with least validation loss
        save_last=False
    )
    callbacks.append(mc)
    trainer = L.Trainer(max_epochs=45,num_nodes=2, callbacks=callbacks) # num_workers, num_nodes?
    trainer.fit(model, train_dataloaders=train_set)
  


    if(dist.get_rank() == 0):
        print("rank 0")
        text = "I don't know when the meeting"
        text = remove_punctuation(text)
        text = text.lower()
        # Load the spaCy English model
        nlp = spacy.load('en_core_web_sm')

        # Tokenize the string using spaCy
        tokens = [token.text for token in nlp(text)]

        print("Original String:", text)
        print("Tokens:", tokens)


        prompt_list = []
        print("vocab", vocab)
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

        # print("rank0 prompt:",prompt)
        # softmax = nn.Softmax(dim=0) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns
        reverse_vocab = {index: word for word, index in vocab.items()}

        # for param in model.parameters():
        #     print("average gradient ramk0:", param)
        model.eval()
        with torch.no_grad():
            for i in range(5):
                result = model(prompt)
                # print("rank0:",result)
                indices = torch.argmax(result, dim=1)
                prompt_list.append(F.one_hot(torch.tensor(indices[len(indices)-1]), len(vocab)).to(dtype=torch.float))
                prompt_stack = torch.stack(prompt_list)
                prompt = torch.unsqueeze(prompt_stack, dim=0)
                print("rank1:",reverse_vocab[indices[len(indices)-1].item()], end= " ")
            print()
    else:
        print("rank 1")
        text = "I don't know when the meeting"
        text = remove_punctuation(text)
        text = text.lower()
        # Load the spaCy English model
        nlp = spacy.load('en_core_web_sm')

        # Tokenize the string using spaCy
        tokens = [token.text for token in nlp(text)]

        print("Original String:", text)
        print("Tokens:", tokens)


        prompt_list = []
        print("vocab", vocab)
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
        # print("rank1 prompt:",prompt)


        # softmax = nn.Softmax(dim=0) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns
        reverse_vocab = {index: word for word, index in vocab.items()}

        model.eval()
        with torch.no_grad():
            for i in range(5):
                result = model(prompt)
                # print("rank1:",result)
                indices = torch.argmax(result, dim=1)
                prompt_list.append(F.one_hot(torch.tensor(indices[len(indices)-1]), len(vocab)).to(dtype=torch.float))
                prompt_stack = torch.stack(prompt_list)
                prompt = torch.unsqueeze(prompt_stack, dim=0)
                print("rank1:",reverse_vocab[indices[len(indices)-1].item()], end= " ")
            print()
        # for param in model.parameters():
        #     print("average gradient rank1:", param)
 

def new_run(rank, size):
    try:
        run(rank, size)
        

    except FileNotFoundError as e:
        print(f"Caught FileNotFoundError: {e}")
        print("Ignoring the error and continuing...")


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4956'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    rank = int(sys.argv[1])
    p = mp.Process(target=init_process, args=(rank, size, run))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()