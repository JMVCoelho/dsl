import torch
from torch import nn
import torchtext
import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from functools import partial
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)

#use env: dsl-2-old

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
configure_seed(seed=42)

class ModelHyperparameters:
    def __init__(
        self,
        n_epochs: int,
        batch_size: int,
        hidden_size: int ,
        learning_rate: float,
        l2: float ,
        teacher_forcing_type: str,
        teacher_forcing_value: float,
        dropout: float,
        reverse_in: bool,
        bidirectional_encoder: bool,
        decoder_attention: bool):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.teacher_forcing_type = teacher_forcing_type
        self.teacher_forcing_value = teacher_forcing_value
        self.dropout = dropout
        self.reverse_in = reverse_in
        self.bidirectional_encoder = bidirectional_encoder
        self.decoder_attention = decoder_attention

def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training models.")
    parser.add_class_arguments(ModelHyperparameters, "training_args")
    parser.add_argument("--cfg",action=ActionConfigFile) 
    return parser 

def tokenize_char(reverse, text):
    """
    Tokenizes a string into individual chars. Can be reversed.
    """
    return [*text][::-1] if reverse else [*text]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.W = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        # This approach like the slides pseudo-code achieved worse results:
        # Will be using the above which considers a single linear layer which receives the concatenation of decoder+encoder info
        #self.W = nn.Linear(hidden_size * 2, hidden_size)
        #self.U = nn.Linear(hidden_size * 1, hidden_size)
        #self.v = nn.Linear(hidden_size, 1)
        

    def forward(self, decoder_hidden, encoder_outputs, mask):
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.shape[0], 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        scores = torch.tanh(self.W(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        #scores = torch.tanh(self.W(encoder_outputs) + self.U(decoder_hidden))
        scores = self.v(scores).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask.T, float('-inf'))
        
        return nn.functional.softmax(scores, dim=0)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, dropout):
        super(LSTMEncoder, self).__init__()
        
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=bidirectional) 

        self.linear_hidden = nn.Linear(hidden_size * 2, hidden_size) if bidirectional else None
        self.linear_cell = nn.Linear(hidden_size * 2, hidden_size) if bidirectional else None

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        output, (hn, cn) = self.lstm(self.dropout(self.embedding(X)))
        mask = None

        if self.bidirectional:
            # last hidden of fwd lstm + last hidden of bwd lstm
            hn = self.linear_hidden(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1))
            cn = self.linear_cell(torch.cat((cn[-2,:,:], cn[-1,:,:]), dim = 1))
            mask = (X == label_field.vocab.stoi[label_field.pad_token])

        return output, hn, cn, mask

class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, attention, bidirectional_encoder, dropout):
        super(LSTMDecoder, self).__init__()
        
        self.output_size = output_size
        
        self.attention = Attention(hidden_size) if attention else None

        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size*3 if bidirectional_encoder else hidden_size, 
                            hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)

        self.dropout = nn.Dropout(dropout)
                
    def forward(self, X, h, c, encoder_outputs, mask):
                        
        embedded = self.dropout(self.embedding(X.unsqueeze(0)))

        if not self.attention:
            output, (hn, cn) = self.lstm(embedded, (h, c))   
            return self.linear(output.squeeze(0)), hn, cn
                          
        else:        
            attention_w = torch.einsum('bij,bjk->bik', self.attention(h, encoder_outputs, mask).unsqueeze(1), encoder_outputs.permute(1,0,2))
            
            # (attn, embed) achieved better results than (embed, attn)
            #lstm_input = torch.cat((embedded, attention_w.permute(1, 0, 2)), dim=2)
            lstm_input = torch.cat((attention_w.permute(1, 0, 2), embedded), dim=2)
            
            output, (hn, cn) = self.lstm(lstm_input, (h.unsqueeze(0), c.unsqueeze(0)))

            return self.linear(output.squeeze(0)), hn.squeeze(0), cn.squeeze(0)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, bidirectional_encoder, decoder_attention):
        super(Seq2SeqLSTM, self).__init__()
        
        self.encoder = LSTMEncoder(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional_encoder, dropout=dropout)
        self.decoder = LSTMDecoder(output_size=output_size, hidden_size=hidden_size, attention=decoder_attention, bidirectional_encoder=bidirectional_encoder, dropout=dropout)

    def forward(self, X, y, teacher_forcing):
        outputs = torch.zeros(y.size(0), X.size(1), self.decoder.output_size)
        encoder_outputs, hidden, cell, mask = self.encoder(X)

        next_token_logits = y[0,:]

        for t in range(1, y.size(0)):
            output, hidden, cell = self.decoder(next_token_logits, hidden, cell, encoder_outputs, mask)
            outputs[t] = output
            next_token_logits = y[t] if random.random() < teacher_forcing else output.argmax(1) 
        
        return outputs


def train_batch(X, y, model, optimizer, criterion, teacher_forcing, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """

    optimizer.zero_grad()
    logits = model(X, y, teacher_forcing=teacher_forcing)

    
    loss = criterion(logits[1:].view(-1, logits.size(-1)), y[1:].view(-1))
    loss.backward()
    # use gradient clipping to mitigate exploding grads
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    return loss

def decode_sequence(sequence):

    special_tokens = [text_field.pad_token, text_field.eos_token, text_field.init_token, text_field.unk_token]
    chars = [[text_field.vocab.itos[idx] for idx in seq.tolist()] for seq in sequence.T]
    
    chars = [[c for c in seq if c not in special_tokens] for seq in chars]
    
    sentence = ["".join(char) for char in chars]
    
    return sentence

def evaluate(model, dataloader):
    n_correct=0
    n_possible=0
    losses = []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X, y, teacher_forcing=0)

            loss_f = nn.CrossEntropyLoss(ignore_index=label_field.vocab.stoi[label_field.pad_token])

            loss = loss_f(logits[1:].view(-1, logits.size(-1)), y[1:].view(-1))

            greedy_choice = logits.argmax(dim=-1)
            
            greedy_words = decode_sequence(greedy_choice)
            label_word = decode_sequence(y)

            n_correct += sum([generation == label for generation, label in zip(greedy_words, label_word)])
            n_possible += float(y.size(1))
            
            losses.append(loss)

    model.train()
   
    return n_correct / n_possible, torch.tensor(losses).mean().item()

def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def decay(initial_prob, num_steps, end_prob=0.3, percentage_fixed=0.3):
    # first k% steps teacher forcing prob is fixed, then starts decaying
    fix_steps = int(num_steps * percentage_fixed) - 1
    decay_steps = num_steps - fix_steps 

    fix_probabilities = [initial_prob] * fix_steps  

    decay_step_size = (initial_prob - end_prob) / (decay_steps - 1)
    decay_probabilities = [initial_prob - decay_step_size * i for i in range(decay_steps)]
    probabilities = fix_probabilities + decay_probabilities
    return probabilities


parser = read_arguments()
cfg = parser.parse_args(["--cfg", "configuration.yaml"])

n_epochs = cfg.training_args.n_epochs
batch_size = cfg.training_args.batch_size
hidden_size = cfg.training_args.hidden_size
learning_rate = cfg.training_args.learning_rate
l2 = cfg.training_args.l2
teacher_forcing_type = cfg.training_args.teacher_forcing_type
teacher_forcing_value = cfg.training_args.teacher_forcing_value
dropout = cfg.training_args.dropout
reverse_in = cfg.training_args.reverse_in
bidirectional_encoder = cfg.training_args.bidirectional_encoder
decoder_attention = cfg.training_args.decoder_attention


if teacher_forcing_type == 'fixed':
    teacher_forcing_probabilities = [teacher_forcing_value] * n_epochs

elif teacher_forcing_type == 'decay':
    teacher_forcing_probabilities = decay(teacher_forcing_value, n_epochs)

else:
    print("err: Available TF types: fixed or decay.")
    exit()


text_field = torchtext.legacy.data.Field(tokenize = partial(tokenize_char, reverse_in), 
            init_token = '<bos>', 
            eos_token = '<eos>', 
            lower = True)

label_field = torchtext.legacy.data.Field(tokenize = partial(tokenize_char, False), 
            init_token = '<bos>', 
            eos_token = '<eos>', 
            lower = True)

fields = [('text', text_field), ('label', label_field)]

train_data, test_data, val_data = torchtext.legacy.data.TabularDataset.splits(
                            path='../../data', 
                            train="ar2en-train.txt",
                            test="ar2en-test.txt", 
                            validation="ar2en-eval.txt",
                            format='tsv', 
                            fields=fields, 
                            skip_header=False)

text_field.build_vocab(train_data, min_freq = 1)
label_field.build_vocab(train_data, min_freq = 1)

# Question 2.1: 
# print(f'source: {len(text_field.vocab)}')
# print(f'target: {len(label_field.vocab)}')
# source: 51
# target: 43

# Field and TabularDataset are legacy... Should change it if I have the time.
# This function maps the output of those functions to something that can be fed to a dataloader.
def collator(batch):
    text_batch = [example.text for example in batch]
    label_batch = [example.label for example in batch]
    text_batch = text_field.process(text_batch)
    label_batch = label_field.process(label_batch)
    return text_batch, label_batch

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collator)
dev_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collator)

model = Seq2SeqLSTM(input_size=len(text_field.vocab), 
                    output_size=len(label_field.vocab), 
                    hidden_size=hidden_size, 
                    dropout=dropout,
                    bidirectional_encoder=bidirectional_encoder, 
                    decoder_attention=decoder_attention)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

criterion = nn.CrossEntropyLoss(ignore_index=label_field.vocab.stoi[label_field.pad_token])

epochs = torch.arange(1, n_epochs + 1)
train_mean_losses = []
valid_accs = []
valid_losses = []
train_losses = []
for ii in epochs:
    print('Training epoch {}'.format(ii))
    p_teacher_forcing = teacher_forcing_probabilities[ii-1]
    print(f'Teacher forcing probability: {p_teacher_forcing}')
    for X_batch, y_batch in tqdm(train_dataloader):
        loss = train_batch(
            X_batch, y_batch, model, optimizer, criterion, p_teacher_forcing)
        train_losses.append(loss)

    mean_loss = torch.tensor(train_losses).mean().item()
    print('Training loss: %.4f' % (mean_loss))

    train_mean_losses.append(mean_loss)
    valid_accuracy, valid_loss = evaluate(model, dev_dataloader)
    print('Valid acc: %.4f' % (valid_accuracy))
    if len(valid_accs)<1 or valid_accuracy>max(valid_accs):
        path_to_model = 'Seq2SeqLSTM_epoch_'+str(ii)+'.pt'
        torch.save({
            'epoch': ii,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
            }, path_to_model)
    valid_accs.append(valid_accuracy)
    valid_losses.append(valid_loss)
## load best model
checkpoint = torch.load(path_to_model)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   

test_acc, test_loss = evaluate(model, test_dataloader)
print(f'Final Test acc: {test_acc}')
# plot
plot(epochs, train_mean_losses, ylabel='Train Loss', name='plots/{}-training-loss-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format('Seq2SeqLSTM', n_epochs, batch_size, hidden_size, learning_rate, l2, teacher_forcing_type, reverse_in, bidirectional_encoder, dropout))
plot(epochs, valid_accs, ylabel='Validation Accuracy', name='plots/{}-validation-accuracy-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format('Seq2SeqLSTM', n_epochs, batch_size, hidden_size, learning_rate, l2, teacher_forcing_type, reverse_in, bidirectional_encoder, dropout))
plot(epochs, valid_losses, ylabel='Validation Loss', name='plots/{}-validation-loss-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format('Seq2SeqLSTM', n_epochs, batch_size, hidden_size, learning_rate, l2, teacher_forcing_type, reverse_in, bidirectional_encoder, dropout))

# Replicate results:
# 2b: config.yaml -> False False False
# 2c: config.yaml -> True False False
# 2d: config.yaml -> False True True

