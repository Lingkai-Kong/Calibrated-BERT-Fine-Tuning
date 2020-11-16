import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from transformers  import BertTokenizer
from transformers  import BertForSequenceClassification
import random
from sklearn.metrics import f1_score
from utils import *
import os 
import argparse



import warnings
warnings.filterwarnings("ignore")

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids, token_type_ids, attention_mask):
        logits = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, args):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECE().to(args.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                batch = tuple(t.to(args.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                logits_list.append(logits)
                labels_list.append(b_labels)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
    
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
 
    
class ECE_v2(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE_v2, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=softmaxes.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
    
def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def main():

    parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
    parser.add_argument('--eva_iter', default=1, type=int, help='number of passes for mc-dropout when evaluation')
    parser.add_argument('--model', type=str, choices=['base', 'manifold-smoothing', 'mc-dropout','temperature'], default='base')
    parser.add_argument('--seed', type=int, default=0, help='random seed for test')
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument('--index', type=int, default=0, help='random seed you used during training')
    parser.add_argument('--in_dataset', required=True, help='target dataset: 20news')
    parser.add_argument('--out_dataset', required=True, help='out-of-dist dataset')
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--saved_dataset', type=str, default='n')
    parser.add_argument('--eps_out', default=0.001, type=float, help="Perturbation size of out-of-domain adversarial training")
    parser.add_argument("--eps_y", default=0.1, type=float, help="Perturbation size of label")
    parser.add_argument('--eps_in', default=0.0001, type=float, help="Perturbation size of in-domain adversarial training")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    outf = 'test/'+args.model+'-'+str(args.index)
    if not os.path.isdir(outf):
        os.makedirs(outf)

    if args.model == 'base':
        dirname = '{}/BERT-base-{}'.format(args.in_dataset, args.index)
        pretrained_dir = './model_save/{}'.format(dirname)
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)
        print('Load Tekenizer')

    elif args.model == 'mc-dropout':
        dirname = '{}/BERT-base-{}'.format(args.in_dataset, args.index)
        pretrained_dir = './model_save/{}'.format(dirname)
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)

    elif args.model == 'temperature':
        dirname = '{}/BERT-base-{}'.format(args.in_dataset, args.index)
        pretrained_dir = './model_save/{}'.format(dirname)
        orig_model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        orig_model.to(args.device)
        model = ModelWithTemperature(orig_model)
        model.to(args.device)
    
    elif args.model == 'manifold-smoothing':
        dirname = '{}/BERT-mf-{}-{}-{}-{}'.format(args.in_dataset, args.index, args.eps_in, args.eps_y, args.eps_out)
        print(dirname)
        pretrained_dir = './model_save/{}'.format(dirname)
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)
        
    
    if args.saved_dataset == 'n':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args.in_dataset)
        _, _, nt_test_sentences, _, _, nt_test_labels = load_dataset(args.out_dataset)

        val_input_ids = []
        test_input_ids = []
        nt_test_input_ids = []

        if args.in_dataset == '20news' or args.in_dataset == '20news-15':
            MAX_LEN = 150
        else:
            MAX_LEN = 256

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                truncation= True,
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        # Add the encoded sentence to the list.
            val_input_ids.append(encoded_sent)


        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                truncation= True,
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        # Add the encoded sentence to the list.
            test_input_ids.append(encoded_sent)

        for sent in nt_test_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                    
                                add_special_tokens = True, 
                                truncation= True,
                                max_length = MAX_LEN,        
                        )
            nt_test_input_ids.append(encoded_sent)
            
        # Pad our input tokens
        val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        nt_test_input_ids = pad_sequences(nt_test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        val_attention_masks = []
        test_attention_masks = []
        nt_test_attention_masks = []

        for seq in val_input_ids:
            seq_mask = [float(i>0) for i in seq]
            val_attention_masks.append(seq_mask)
        for seq in test_input_ids:
            seq_mask = [float(i>0) for i in seq]
            test_attention_masks.append(seq_mask)
        for seq in nt_test_input_ids:
            seq_mask = [float(i>0) for i in seq]
            nt_test_attention_masks.append(seq_mask)


        val_inputs = torch.tensor(val_input_ids)
        val_labels = torch.tensor(val_labels)
        val_masks = torch.tensor(val_attention_masks)

        test_inputs = torch.tensor(test_input_ids)
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(test_attention_masks)

        nt_test_inputs = torch.tensor(nt_test_input_ids)
        nt_test_labels = torch.tensor(nt_test_labels)
        nt_test_masks = torch.tensor(nt_test_attention_masks)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        nt_test_data = TensorDataset(nt_test_inputs, nt_test_masks, nt_test_labels)
        
        dataset_dir = 'dataset/test'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        torch.save(val_data, dataset_dir+'/{}_val_in_domain.pt'.format(args.in_dataset))
        torch.save(test_data, dataset_dir+'/{}_test_in_domain.pt'.format(args.in_dataset))
        torch.save(nt_test_data, dataset_dir+'/{}_test_out_of_domain.pt'.format(args.out_dataset))
    
    else:
        dataset_dir = 'dataset/test'
        val_data = torch.load(dataset_dir+'/{}_val_in_domain.pt'.format(args.in_dataset))
        test_data = torch.load(dataset_dir+'/{}_test_in_domain.pt'.format(args.in_dataset))
        nt_test_data = torch.load(dataset_dir+'/{}_test_out_of_domain.pt'.format(args.out_dataset))
        
        

    
    
######## saved dataset
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    nt_test_sampler = SequentialSampler(nt_test_data)
    nt_test_dataloader = DataLoader(nt_test_data, sampler=nt_test_sampler, batch_size=args.eval_batch_size)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size)
    
    if args.model == 'temperature':
        model.set_temperature(val_dataloader, args)
    
    model.eval()
    
    if args.model == 'mc-dropout':
        model.apply(apply_dropout)

    correct = 0
    total = 0
    output_list = []
    labels_list = []
 
##### validation dat
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            total += b_labels.shape[0]
            batch_output = 0
            for j in range(args.eva_iter):
                if args.model == 'temperature':
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask) #logits
                else:
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]  #logits
                batch_output = batch_output + F.softmax(current_batch, dim=1)
            batch_output = batch_output/args.eva_iter
            output_list.append(batch_output)
            labels_list.append(b_labels)
            score, predicted = batch_output.max(1)
            correct += predicted.eq(b_labels).sum().item()

    ###calculate accuracy and ECE
    val_eval_accuracy = correct/total
    print("Val Accuracy: {}".format(val_eval_accuracy))
    ece_criterion = ECE_v2().to(args.device)
    softmaxes_ece = torch.cat(output_list)
    labels_ece = torch.cat(labels_list)
    val_ece = ece_criterion(softmaxes_ece, labels_ece).item()
    print('ECE on Val data: {}'.format(val_ece))

#### Test data
    correct = 0
    total = 0
    output_list = []
    labels_list = []
    predict_list = []
    true_list = []
    true_list_ood = []
    predict_mis = []
    predict_in = []
    score_list = []
    correct_index_all = []
    ## test on in-distribution test set
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            total += b_labels.shape[0]
            batch_output = 0
            for j in range(args.eva_iter):
                if args.model == 'temperature':
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask) #logits
                else:
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]  #logits
                batch_output = batch_output + F.softmax(current_batch, dim=1)
            batch_output = batch_output/args.eva_iter
            output_list.append(batch_output)
            labels_list.append(b_labels)
            score, predicted = batch_output.max(1)
     
            correct += predicted.eq(b_labels).sum().item()
            
            correct_index = (predicted == b_labels)
            correct_index_all.append(correct_index) 
            score_list.append(score)
    
    ###calcutae accuracy 
    eval_accuracy = correct/total
    print("Test Accuracy: {}".format(eval_accuracy))
    
    ##calculate ece
    ece_criterion = ECE_v2().to(args.device)
    softmaxes_ece = torch.cat(output_list)
    labels_ece = torch.cat(labels_list)
    ece = ece_criterion(softmaxes_ece, labels_ece).item()
    print('ECE on Test data: {}'.format(ece))

    #confidence for in-distribution data
    score_in_array = torch.cat(score_list)
    #indices of data that are classified correctly
    correct_array = torch.cat(correct_index_all)
    label_array = torch.cat(labels_list)

### test on out-of-distribution data
    predict_ood = []
    score_ood_list = []
    true_list_ood = []
    with torch.no_grad():
        for step, batch in enumerate(nt_test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            batch_output = 0
            for j in range(args.eva_iter):
                if args.model == 'temperature':
                    current_batch = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                else:
                    current_batch = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                batch_output = batch_output + F.softmax(current_batch, dim=1)
            batch_output = batch_output/args.eva_iter
            score_out, _ = batch_output.max(1)  

            score_ood_list.append(score_out)
            
    score_ood_array = torch.cat(score_ood_list)
    
    
    
    label_array = label_array.cpu().numpy()
    score_ood_array = score_ood_array.cpu().numpy()
    score_in_array = score_in_array.cpu().numpy()
    correct_array = correct_array.cpu().numpy()
    
    
    
    
 ####### calculate NBAUCC for detection task   
    predict_o = np.zeros(len(score_in_array)+len(score_ood_array)) 
    true_o = np.ones(len(score_in_array)+len(score_ood_array))
    true_o[:len(score_in_array)] = 0   ## in-distribution data as false, ood data as positive
    true_mis = np.ones(len(score_in_array))
    true_mis[correct_array] = 0  ##true instances as false, misclassified instances as positive
    predict_mis = np.zeros(len(score_in_array))
    


    ood_sum = 0
    mis_sum = 0
    
    ood_sum_list = []
    mis_sum_list = []

#### upper bound of the threshold tau for NBAUCC
    stop_points = [0.50, 1.]

    for threshold in np.arange(0., 1.01, 0.02):
        predict_ood_index1 = (score_in_array < threshold)
        predict_ood_index2 = (score_ood_array < threshold)
        predict_ood_index = np.concatenate((predict_ood_index1, predict_ood_index2), axis=0)
        predict_o[predict_ood_index] = 1 
        predict_mis[score_in_array<threshold] = 1
        
        ood = f1_score(true_o, predict_o, average='binary') ##### detection f1 score for a specific threshold 
        mis = f1_score(true_mis, predict_mis, average='binary')


        ood_sum += ood*0.02
        mis_sum += mis*0.02
        
        if threshold in stop_points:
            ood_sum_list.append(ood_sum)
            mis_sum_list.append(mis_sum)
            
    for i in range(len(stop_points)):
        print('OOD detection, NBAUCC {}: {}'.format(stop_points[i], ood_sum_list[i]/stop_points[i]))
        print('misclassification detection, NBAUCC {}: {}'.format(stop_points[i], mis_sum_list[i]/stop_points[i]))

if __name__ == "__main__":
    main()
