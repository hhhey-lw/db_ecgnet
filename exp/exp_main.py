import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from .exp_basic import Exp_Basic
from data.dataset import load_datasets
from utils.metrics import metric

import models


class Exp_Main(Exp_Basic):
    def _build_model(self, num_classes):
        return getattr(models, self.args.model)(12, num_classes)

    def _get_data(self):
        train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(datafolder=self.args.datafolder,
                                                                                       experiment=self.args.experiment,
                                                                                       batch_size=self.args.batch_size,
                                                                                       sampling_frequency=self.args.sampling_frequency)
        return train_dataloader, val_dataloader, test_dataloader, num_classes

    def _select_optimizer(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-3)
        return optimizer

    def _select_criterion(self):
        if self.args.is_multi_label:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_scheduler(self, optimizer):
        scheduler = MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.step_gamma)
        return scheduler

    def compute_metric(self, phase, loss, y_pred, y_true):
        if self.args.is_multi_label:
            auc, f1, acc, recall, precision, mAP = metric(y_pred, y_true, self.args.is_multi_label)
            print('%s_loss: %.4f, macro_auc: %.4f, f1: %.4f, acc: %.4f, recall: %.4f, precision: %4f, mAP: %.4f' %
                  (phase, loss, auc, f1, acc, recall, precision, mAP))
            return loss, auc, f1, acc, recall, precision, mAP
        else:
            auc, f1, acc, recall, precision = metric(y_pred, y_true, self.args.is_multi_label)
            print('%s_loss: %.4f, macro_auc: %.4f, f1: %.4f, acc: %.4f, recall: %.4f, precision: %.4f' %
                  (phase, loss, auc, f1, acc, recall, precision))
            return loss, auc, f1, acc, recall, precision

    def save_metric_result(self, epoch, train_metric, val_metric, test_metric):
        result_list = [[epoch, *train_metric, *val_metric, *test_metric]]

        # record metrics
        if epoch == 1:
            if self.args.is_multi_label:
                columns = ['epoch',
                           'train_loss', 'train_auc', 'train_f1', 'train_acc', 'train_recall', 'train_precision', 'train_mAP',
                           'val_loss', 'val_auc', 'val_f1', 'val_acc', 'val_recall', 'val_precision', 'val_mAP',
                           'test_loss', 'test_auc', 'test_f1', 'test_acc', 'test_recall', 'test_precision', 'test_mAP']
            else:
                columns = ['epoch',
                           'train_loss', 'train_auc', 'train_f1', 'train_acc', 'train_recall', 'train_precision',
                           'val_loss', 'val_auc', 'val_f1', 'val_acc', 'val_recall', 'val_precision',
                           'test_loss', 'test_auc', 'test_f1', 'test_acc', 'test_recall', 'test_precision']

            dt = pd.DataFrame(result_list, columns=columns)
            dt.to_csv(self.args.records_path, index=False)

        else:
            dt = pd.DataFrame(result_list)
            dt.to_csv(self.args.records_path, mode='a', header=False, index=False)

    def save_model_weight(self, best, val_metric):
        if self.args.is_multi_label:
            val_loss, val_auc, val_f1, val_acc, val_recall, val_precision, val_mAP = val_metric
            if best < val_f1:  # You can choose other metric
                best = val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, self.args.checkpoints_best)
        else:
            val_loss, val_auc, val_f1, val_acc, val_recall, val_precision = val_metric
            if best < val_f1:
                best = val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, self.args.checkpoints_best)
        return best

    def train(self):
        train_dataloader, val_dataloader, test_dataloader, num_classes = self._get_data()
        self.model = self._build_model(num_classes).to(self.device)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(optimizer)
        best = .0
        for epoch in range(1, self.args.train_epochs + 1):
            current_lr = scheduler.get_last_lr()[0]

            print(f'@ epoch: {epoch}, learning rate: {np.round(current_lr, decimals=8)}')

            train_metric = self._train_one_epoch(train_dataloader, criterion, optimizer)
            val_metric = self.vali('vali', val_dataloader, criterion)
            test_metric = self.vali('test', test_dataloader, criterion)

            scheduler.step()
            self.save_metric_result(epoch, train_metric, val_metric, test_metric)
            best = self.save_model_weight(best, val_metric)

    def vali(self, phase, dataloader, criterion):
        loss_meter, it_count = 0, 0
        pred_arr = []
        true_arr = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                loss_meter += loss.item()
                it_count += 1
                if self.args.is_multi_label:
                    y_prob = torch.sigmoid(outputs)
                else:
                    y_prob = F.softmax(outputs, dim=-1)
                y_true = batch_y
                for i in range(len(y_prob)):
                    pred_arr.append(y_prob[i].cpu().detach().numpy())
                    true_arr.append(y_true[i].cpu().detach().numpy())

        return self.compute_metric(phase, loss_meter / it_count, pred_arr, true_arr)

    def test(self):
        _, _, test_dataloader, num_classes = self._get_data()
        self.model = self._build_model(num_classes).to(self.device)
        ckpt = torch.load(self.args.checkpoints_best, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        pred_arr = []
        true_arr = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_dataloader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)

                if self.args.is_multi_label:
                    y_prob = torch.sigmoid(outputs)
                else:
                    y_prob = F.softmax(outputs, dim=-1)
                y_true = batch_y
                for i in range(len(y_prob)):
                    pred_arr.append(y_prob[i].cpu().detach().numpy())
                    true_arr.append(y_true[i].cpu().detach().numpy())

        if self.args.is_multi_label:
            auc, f1, acc, recall, precision, mAP = metric(pred_arr, true_arr, self.args.is_multi_label)
            print('macro_auc: %.4f, f1: %.4f, acc: %.4f, recall: %.4f, precision: %4f, mAP: %.4f'
                  % (auc, f1, acc, recall, precision, mAP))
        else:
            auc, f1, acc, recall, precision = metric(pred_arr, true_arr, self.args.is_multi_label)
            print('macro_auc: %.4f, f1: %.4f, acc: %.4f, recall: %.4f, precision: %4f'
                  % (auc, f1, acc, recall, precision))

    def _train_one_epoch(self, dataloader, criterion, optimizer):
        loss_meter, it_count = 0, 0
        pred_arr = []
        true_arr = []
        self.model.train()
        for batch_x, batch_y in tqdm(dataloader):
            batch_x += torch.randn_like(batch_x) * 0.1  # Data Augment
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            loss_meter += loss.item()
            it_count += 1
            if self.args.is_multi_label:
                y_prob = torch.sigmoid(outputs)
            else:
                y_prob = F.softmax(outputs, dim=-1)
            y_true = batch_y
            for i in range(len(y_prob)):
                pred_arr.append(y_prob[i].cpu().detach().numpy())
                true_arr.append(y_true[i].cpu().detach().numpy())

        return self.compute_metric('train', loss_meter / it_count, pred_arr, true_arr)
