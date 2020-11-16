import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers  import BertTokenizer, BertConfig
from transformers  import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients
import argparse
import random
from utils import *
import os 


class ECE(nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

# Function to calculate the accuracy of our predictions vs labels
def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='20news-15', type=str, help="dataset")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--saved_dataset', type=str, default='n', help='whether save the preprocessed pt file of the dataset')

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    ece_criterion = ECE().to(args.device)

    # load dataset
    if args.saved_dataset == 'n':
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args.dataset)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        train_input_ids = []
        val_input_ids = []
        test_input_ids = []
        
        if args.dataset == '20news' or args.dataset == '20news-15':
            MAX_LEN = 150
        else:
            MAX_LEN = 256

        for sent in train_sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
            encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        # Add the encoded sentence to the list.
            train_input_ids.append(encoded_sent)
            

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                    
                                add_special_tokens = True, 
                                max_length = MAX_LEN,         
                        )
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                     
                                add_special_tokens = True, 
                                max_length = MAX_LEN,        
                        )
            test_input_ids.append(encoded_sent)

        # Pad our input tokens
        train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        train_attention_masks = []
        val_attention_masks = []
        test_attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in train_input_ids:
            seq_mask = [float(i>0) for i in seq]
            train_attention_masks.append(seq_mask)
        for seq in val_input_ids:
            seq_mask = [float(i>0) for i in seq]
            val_attention_masks.append(seq_mask)
        for seq in test_input_ids:
            seq_mask = [float(i>0) for i in seq]
            test_attention_masks.append(seq_mask)

        # Convert all of our data into torch tensors, the required datatype for our model

        train_inputs = torch.tensor(train_input_ids)
        validation_inputs = torch.tensor(val_input_ids)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(val_labels)
        train_masks = torch.tensor(train_attention_masks)
        validation_masks = torch.tensor(val_attention_masks)
        test_inputs = torch.tensor(test_input_ids)
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(test_attention_masks)

        # Create an iterator of our data with torch DataLoader. 
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
        
        dataset_dir = 'dataset/{}'.format(args.dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        torch.save(train_data, dataset_dir+'/train.pt')
        torch.save(validation_data, dataset_dir+'/val.pt')
        torch.save(prediction_data, dataset_dir+'/test.pt')

    else:
        dataset_dir = 'dataset/{}'.format(args.dataset)
        train_data = torch.load(dataset_dir+'/train.pt')
        validation_data = torch.load(dataset_dir+'/val.pt')
        prediction_data = torch.load(dataset_dir+'/test.pt')


    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)

    

    if args.dataset == '20news':
        num_labels = 20
    elif args.dataset == '20news-15':
        num_labels = 15
    elif args.dataset == 'wos-100':
        num_labels = 100
    elif args.dataset == 'wos':
        num_labels = 134

    print(num_labels)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels= num_labels, output_hidden_states=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(args.device)

#######train model 

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    t_total = len(train_dataloader) * args.epochs
    # Store our loss and accuracy for plotting

    best_val = -np.inf
    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(args.epochs, desc="Epoch"): 
    # Training
        # Set our model to training mode (as opposed to evaluation mode)
        # Tracking variables
        tr_loss =  0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            loss_ce = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            if torch.cuda.device_count() > 1:
                loss_ce = loss_ce.mean()
            loss_ce.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss_ce.item()
        
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train cross entropy loss: {}".format(tr_loss/nb_tr_steps))
        
    
        
        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_eval_examples = 0
        logits_list = []
        labels_list = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0] 
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
    
            eval_accurate_nb += tmp_eval_nb
            nb_eval_examples += label_ids.shape[0]
        eval_accuracy = eval_accurate_nb/nb_eval_examples
        print("Validation Accuracy: {}".format(eval_accuracy))
        scheduler.step(eval_accuracy)

        logits_ece = torch.cat(logits_list)
        labels_ece = torch.cat(labels_list)
        ece = ece_criterion(logits_ece, labels_ece).item()
        print('ECE on val data: {}'.format(ece))
        

        if eval_accuracy > best_val:
            dirname = '{}/BERT-base-{}'.format(args.dataset, args.seed)
 
            output_dir = './model_save/{}'.format(dirname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print("Saving model to %s" % output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)   
            #tokenizer.save_pretrained(output_dir)

            best_val = eval_accuracy

# ##### test model on test data
        # Put model in evaluation mode
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_test_examples = 0
        logits_list = []
        labels_list = []
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
            eval_accurate_nb += tmp_eval_nb
            nb_test_examples += label_ids.shape[0]

        print("Test Accuracy: {}".format(eval_accurate_nb/nb_test_examples))

        logits_ece = torch.cat(logits_list)
        labels_ece = torch.cat(labels_list)
        ece = ece_criterion(logits_ece, labels_ece).item()

        print('ECE on test data: {}'.format(ece))
        

if __name__ == "__main__":
    main()