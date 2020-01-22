import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import csv
import random
from utils import data_reader
from utils import config
import os
import matplotlib.pyplot as plt

LABELS = {
    "OFF": 1,
    "NOT": 0,
}


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class ClassificationModel:
    def __init__(self, bert_model=config.bert_model, gpu=False, seed=0):

        self.gpu = gpu
        self.bert_model = bert_model

        self.train_df, self.test_df, self.val_df  = data_reader.load_dataset(config.data_path)

        self.num_classes = len(LABELS)

        self.model = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        # to plot loss during training process
        self.plt_x = []
        self.plt_y = []


        # to plot loss during training process
        self.plt_x_l = []
        self.plt_y_l = []


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)

    def __init_model(self):
        if self.gpu:
            self.device = torch.device("cuda")
            # print(torch.cuda.memory_allocated(self.device))
            # # log available cuda
            # if self.device.type == 'cuda':
            #     print(torch.cuda.get_device_name(0))
            #     print('Memory Usage:')
            #     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            #     print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def new_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_classes,output_attentions = False, output_hidden_states = True)
        self.__init_model()

    def load_model(self, path_model, path_config):
        # self.model = BertForSequenceClassification(BertConfig(path_config), num_labels=self.num_classes,output_attentions = False, output_hidden_states = True)
        self.model = BertForSequenceClassification.from_pretrained(path_model)
        self.tokenizer = BertTokenizer.from_pretrained(path_model)
        # self.model.load_state_dict(torch.load(path_model))
        self.__init_model()

    def save_model(self, path_model, path_config, epoch_n, acc, f1, ave_loss):

        import os

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

        output_dir = path_model

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model,
                                                'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


        # if not os.path.exists(path_model):
        #     os.makedirs(path_model)
        #
        # model_save_path = os.path.join(path_model,'model_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(epoch_n,ave_loss, acc, f1))
        #
        # torch.save(self.model.state_dict(), model_save_path)
        #
        # if not os.path.exists(path_config):
        #     os.makedirs(path_config)
        #
        # model_config_path = os.path.join(path_model,'config.cf')
        # with open(model_config_path, 'w') as f:
        #     f.write(self.model.config.to_json_string())

    def train(self, epochs, batch_size=config.batch_size, lr=config.lr, plot_path=None , model_path=None, config_path=None):

        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, warmup=0.1,
        #                           t_total=int(len(self.train_df) / batch_size) * epochs)

        self.optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        nb_tr_steps = 0
        train_features = data_reader.convert_examples_to_features(self.train_df, config.MAX_SEQ_LENGTH, self.tokenizer)

        # create tensor of all features
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # class weighting
        _, counts = np.unique(self.train_df['subtask_a'], return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        # assign wight to each input sample
        example_weights = [class_weights[e] for e in self.train_df['subtask_a']]
        sampler = WeightedRandomSampler(example_weights, len(self.train_df['subtask_a']))
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)



        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        for e in range(epochs):
            print(f"Epoch {e}")

            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                self.model.zero_grad()

                outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask, labels= label_ids)
                loss = outputs[0]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                # Update the learning rate.
                scheduler.step()

                total_loss += loss.item()

                if plot_path is not None :
                    self.plt_y.append(loss.item())
                    self.plt_x.append(nb_tr_steps)
                    self.save_plot(plot_path)

                nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.gpu:
                    torch.cuda.empty_cache()

            f1, acc = self.val()
            print(f"\nF1 score: {f1}, Accuracy: {acc}")

            loss = total_loss / len(train_dataloader)
            print("epoch {} loss: {}".format(e,loss))
            if plot_path is not None:
                self.plt_y_l.append(loss)
                self.plt_x_l.append(e)
                self.save_loss_plot2(plot_path+'2')


            if model_path is not None and config_path is not None:
                self.save_model(model_path, config_path, e, acc, f1, loss)


    def val(self, batch_size=config.batch_size):
        eval_features = data_reader.convert_examples_to_features(self.val_df, config.MAX_SEQ_LENGTH, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        f1, acc = 0, 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

            logits = outputs[0]

            predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc += np.sum(predicted_labels == gnd_labels.numpy())
            tmp_eval_f1 = f1_score(predicted_labels, gnd_labels, average='macro')
            f1 += tmp_eval_f1 * input_ids.size(0)
            nb_eval_examples += input_ids.size(0)

        return f1 / nb_eval_examples, acc / nb_eval_examples

    def save_plot(self, path):

        fig, ax = plt.subplots()
        ax.plot(self.plt_x, self.plt_y)

        ax.set(xlabel='Training steps', ylabel='Loss')

        fig.savefig(path)
        plt.close()


    def save_loss_plot2(self,path):
        fig, ax = plt.subplots()
        ax.plot(self.plt_x_l, self.plt_y_l)

        ax.set(xlabel='epoch', ylabel='Loss')

        fig.savefig(path)
        plt.close()


    def create_test_predictions(self, path):


        tests_features = data_reader.convert_examples_to_features(self.test_df,
                                                                 config.MAX_SEQ_LENGTH,
                                                                 self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in tests_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in tests_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in tests_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in tests_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)

        predictions = []
        predictions_to_save = []
        actual_to_save = []
        inverse_labels = {v: k for k, v in LABELS.items()}

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
            logits = outputs[0]
            predictions += [inverse_labels[p] for p in list(np.argmax(logits.detach().cpu().numpy(), axis=1))]
            actual_to_save += gnd_labels.tolist()
            predictions_to_save += list(np.argmax(logits.detach().cpu().numpy(), axis=1))

        return actual_to_save, predictions_to_save

    def create_embedding(self):


        tests_features = data_reader.convert_examples_to_features(self.test_df,
                                                                 config.MAX_SEQ_LENGTH,
                                                                 self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in tests_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in tests_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in tests_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in tests_features], dtype=torch.long)
        all_actual_input_id = torch.tensor([f.actual_input_id for f in tests_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_actual_input_id)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

        embedding_dict = {}
        for input_ids, input_mask, segment_ids, gnd_labels, actual_input_id in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

            embedding_dict.update( dict(zip(actual_input_id.tolist(), outputs[1])) )

        self.test_df['bert'] = self.test_df['id'].map(embedding_dict)
        data_reader.save_obj(self.test_df, 'test_em')


        train_features = data_reader.convert_examples_to_features(self.train_df,
                                                                 config.MAX_SEQ_LENGTH,
                                                                 self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_actual_input_id = torch.tensor([f.actual_input_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_actual_input_id)

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

        embedding_dict = {}
        for input_ids, input_mask, segment_ids, gnd_labels, actual_input_id in tqdm(train_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

            embedding_dict.update( dict(zip(actual_input_id.tolist(), outputs[1])) )

        self.train_df['bert'] = self.train_df['id'].map(embedding_dict)
        data_reader.save_obj(self.train_df, 'train_em')

        eval_features = data_reader.convert_examples_to_features(self.val_df, config.MAX_SEQ_LENGTH, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_actual_input_id = torch.tensor([f.actual_input_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_actual_input_id)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        embedding_dict = {}

        for input_ids, input_mask, segment_ids, gnd_labels, actual_input_id in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            embedding_dict.update(dict(zip(actual_input_id.tolist(), outputs[1])))

        self.val_df['bert'] = self.val_df['id'].map(embedding_dict)
        data_reader.save_obj(self.val_df, 'valid_em')

