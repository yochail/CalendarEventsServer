# use BERT fine tuning for NER
import random

import numpy as np
import pandas as pd
import torch

from pandas import DataFrame
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score
from tqdm import trange

data = pd.read_csv("data/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
print(data.tail(10))



class Token:
	def __init__(self,word,pos,tag):
		self.text = word
		self.pos = pos
		self.tag = tag


def preper_data(data:DataFrame):
		groups = [g[1] for g in data.groupby("Sentence #")]
		text,labels = zip(*[(' '.join(g["Word"].values.tolist()),
		                   ' '.join(g["Tag"].values.tolist()))
		                  for g in groups])
		return text,labels





# preper data for bert
MAX_LEN = 75
batch_size = 32
test_size = 0.1

sentences,labels = preper_data(data)
tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=test_size)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=test_size)
tr_inputs = torch.Tensor(tr_inputs)
val_inputs = torch.Tensor(val_inputs)
tr_tags = torch.Tensor(tr_tags)
val_tags = torch.Tensor(val_tags)
tr_masks = torch.Tensor(tr_masks)
val_masks = torch.Tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

FULL_FINETUNING = False
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = 5
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
	# TRAIN loop
	model.train()
	tr_loss = 0
	nb_tr_examples, nb_tr_steps = 0, 0
	for step, batch in enumerate(train_dataloader):
		# add batch to gpu
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels = batch
		# forward pass
		loss = model(b_input_ids, token_type_ids=None,
		             attention_mask=b_input_mask, labels=b_labels)
		# backward pass
		loss.backward()
		# track train loss
		tr_loss += loss.item()
		nb_tr_examples += b_input_ids.size(0)
		nb_tr_steps += 1
		# gradient clipping
		torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
		# update parameters
		optimizer.step()
		model.zero_grad()
	# print train loss per epoch
	print("Train loss: {}".format(tr_loss / nb_tr_steps))
	# VALIDATION on validation set
	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	predictions, true_labels = [], []
	for batch in valid_dataloader:
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels = batch

		with torch.no_grad():
			tmp_eval_loss = model(b_input_ids, token_type_ids=None,
			                      attention_mask=b_input_mask, labels=b_labels)
			logits = model(b_input_ids, token_type_ids=None,
			               attention_mask=b_input_mask)
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
		true_labels.append(label_ids)

		tmp_eval_accuracy = flat_accuracy(logits, label_ids)

		eval_loss += tmp_eval_loss.mean().item()
		eval_accuracy += tmp_eval_accuracy

		nb_eval_examples += b_input_ids.size(0)
		nb_eval_steps += 1
	eval_loss = eval_loss / nb_eval_steps
	print("Validation loss: {}".format(eval_loss))
	print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
	pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
	valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
	print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))