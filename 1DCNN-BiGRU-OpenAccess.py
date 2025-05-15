# MIT License
#
# Copyright (c) 2025 Alex Zhu / Hangzhou Normal University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

class Config:
    # Original index correspondence:
    # 0: x coordinate 1: y coordinate   2: timestamp    3: x speed 4: y speed
    # 5: x acceleration 6: y acceleration    7: x jerk 8: y jerk

    spiral_selected_columns = [3, 4, 5, 6] # 0-based
    spiral_num_features = 4  # len of selected_columns
    spiral_window_size = 32
    spiral_stride = 8
    spiral_inception_blocks = 2
    spiral_gru_hidden_size = 64
    spiral_threshold = 0.5
    spiral_vote_rate = 0.5

    wavy_selected_columns = [0, 1, 3, 4, 5, 6, 7, 8]  # 0-based
    wavy_num_features = 8  # len of selected_columns
    wavy_window_size = 32
    wavy_stride = 8
    wavy_inception_blocks = 2
    wavy_gru_hidden_size = 64
    wavy_threshold = 0.5
    wavy_vote_rate = 0.5

    num_classes = 1

    # training parameters
    k_folds = 10
    batch_size = 4
    learning_rate = 0.001
    max_epochs = 20
    patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OneDCnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, channel_set=8):
        super().__init__()
        # channel_set = 8

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, channel_set, kernel_size=1, padding='same'),
            nn.BatchNorm1d(channel_set),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, channel_set, kernel_size=3, padding='same'),
            nn.BatchNorm1d(channel_set),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, channel_set, kernel_size=5, padding='same'),
            nn.BatchNorm1d(channel_set),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, channel_set, kernel_size=1),
            nn.BatchNorm1d(channel_set),
            nn.ReLU()
        )

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, 4*channel_set, kernel_size=1),
            nn.BatchNorm1d(4*channel_set)
        ) if in_channels != 4*channel_set else nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        branches = [self.branch1(x), self.branch2(x),
                    self.branch3(x), self.branch4(x)]
        out = torch.cat(branches, dim=1)
        return self.activation(out + residual)

class DiagnosisModel(nn.Module):
    def __init__(self, config, is_spiral):
        super().__init__()
        self.config = config

        if is_spiral:
            in_channels = config.spiral_num_features
            inception_blocks = config.spiral_inception_blocks
            gru_hidden_size = config.spiral_gru_hidden_size
        else:
            in_channels = config.wavy_num_features
            inception_blocks = config.wavy_inception_blocks
            gru_hidden_size = config.wavy_gru_hidden_size

        self.cnn_blocks = nn.Sequential()

        # in_channels = x
        channel_set = 8
        if inception_blocks >= 1: self.cnn_blocks.add_module(f'inception_0', OneDCnnBlock(in_channels, channel_set=channel_set))
        in_channels = 32
        channel_set = 16
        if inception_blocks >= 2: self.cnn_blocks.add_module(f'inception_1', OneDCnnBlock(in_channels, channel_set=channel_set))
        in_channels = 64
        channel_set = 16
        for i in range(inception_blocks-2):
            self.cnn_blocks.add_module(f'inception_{i+2}', OneDCnnBlock(in_channels, channel_set=channel_set))

        self.gru = nn.GRU(input_size=64,
                          hidden_size=gru_hidden_size,
                          bidirectional=True,
                          batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * gru_hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, config.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # input shape: (batch_size, seq_len, num_features)
        x = x.permute(0, 2, 1)  # trans to:  (batch_size, num_features, seq_len)

        x = self.cnn_blocks(x)
        x = x.permute(0, 2, 1)  # trans to: (batch_size, seq_len, features)

        # GRU
        gru_out, _ = self.gru(x)

        # last time step
        out = gru_out[:, -1, :]

        return self.fc(out)

# preprocess
def process_modal_data(folder_path, selected_columns, window_size=32, stride=8, need_last = True):
    samples = {}
    labels = {}

    for label_folder in ['healthy', 'patient']:
        label = 0 if label_folder == 'healthy' else 1
        path = os.path.join(folder_path, label_folder)
        for filename in os.listdir(path):
            if not filename.endswith('.txt'): continue
            file_path = os.path.join(path, filename)
            subj_id = int(filename[:3])
            with open(file_path, 'r') as f:
                data = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    items = line.split(',')
                    if len(items) != 9: continue
                    selected = [float(items[i]) for i in selected_columns]
                    data.append(selected)
                if len(data) < 1: continue
                # sliding window
                data = np.array(data)
                windows = []
                num_windows = (len(data) - window_size) // stride + 1
                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    window = data[start:end]
                    windows.append(window)
                if need_last:
                    # last partial window
                    if len(data) % stride != 0:
                        last_window = np.zeros((window_size, len(selected_columns)))
                        remaining = data[-(len(data) % stride):]
                        last_window[:len(remaining)] = remaining
                        windows.append(last_window)
                # normalize
                windows = np.array(windows)
                for i in range(windows.shape[2]):
                    scaler = StandardScaler()
                    windows[:, :, i] = scaler.fit_transform(windows[:, :, i].reshape(-1, 1)).reshape(windows.shape[0], -1)

                samples[subj_id] = np.array(windows)  # {id, data} , data shape (num_windows , num_features)
                labels[subj_id] = label
    return samples, labels

def main():
    config = Config()

    # print(str(config.spiral_window_size))
    # print(str(config.spiral_stride))
    # print(str(config.wavy_window_size))
    # print(str(config.wavy_stride))

    # load two modals data
    # data: dict{ int, nparray(m,features) }, label: dict{int, List[int]}
    modal1_data, modal1_labels = process_modal_data('../Spiral', config.spiral_selected_columns, config.spiral_window_size, config.spiral_stride)
    modal2_data, modal2_labels = process_modal_data('../Wavy', config.wavy_selected_columns, config.wavy_window_size, config.wavy_stride)

    # list [int]
    valid_subjects_id = sorted(set(modal1_labels.keys()) & set(modal2_labels.keys()))

    # list[int]
    y_true = [modal1_labels[s] for s in valid_subjects_id]  # gold

    results = {s: {'modal1_probs': [], 'modal1_predict': [], 'modal1_votes': [],'modal1_raw_probs': [],
                   'modal2_probs': [],  'modal2_predict': [], 'modal2_votes': [],'modal2_raw_probs': [],
                   'true_label': modal1_labels[s]} for s in valid_subjects_id}

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True)

    all_train_losses_modal_1, all_val_losses_modal_1 = [], []
    all_train_accuracies_modal_1, all_val_accuracies_modal_1 = [], []

    all_train_losses_modal_2, all_val_losses_modal_2 = [], []
    all_train_accuracies_modal_2, all_val_accuracies_modal_2 = [], []

    timestamp = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_subjects_id, y_true)):
        print(f"\n=== Fold {fold + 1} ===")

        this_fold_train_losses_modal_1, this_fold_val_losses_modal_1 = [], []
        this_fold_train_accuracies_modal_1, this_fold_val_accuracies_modal_1 = [], []

        this_fold_train_losses_modal_2, this_fold_val_losses_modal_2 = [], []
        this_fold_train_accuracies_modal_2, this_fold_val_accuracies_modal_2 = [], []

        train_subjects_id = [valid_subjects_id[i] for i in train_idx]  # index to actual id
        val_subjects_id = [valid_subjects_id[i] for i in val_idx]

        modal_1_model = None
        modal_2_model = None

        modal_1_val_loader = None
        modal_2_val_loader = None

        # train two modals
        for cur_modal_str in ['modal1', 'modal2']:
            is_modal_1 = (cur_modal_str == 'modal1')
            curr_data = modal1_data if is_modal_1 else modal2_data
            curr_labels = modal1_labels if is_modal_1 else modal2_labels

            curr_threshod = config.spiral_threshold if is_modal_1 else config.wavy_threshold
            curr_vote_rate = config.spiral_vote_rate if is_modal_1 else config.wavy_vote_rate

            train_windows = []
            train_labels = []
            for subj_id in train_subjects_id:
                windows = curr_data[subj_id]
                train_windows.extend(windows)
                train_labels.extend([windows.shape[0] * [curr_labels[subj_id]]])

            val_windows = []
            val_labels = []
            val_window_subject_map = []
            for subj_id in val_subjects_id:
                windows = curr_data[subj_id]
                val_windows.extend(windows)
                val_labels.extend([windows.shape[0] * [curr_labels[subj_id]]])
                val_window_subject_map.extend(windows.shape[0] * [subj_id])

            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(np.array(train_windows)),
                torch.FloatTensor(np.array([element for sublist in train_labels for element in sublist] ).flatten()))

            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(np.array(val_windows)),
                torch.FloatTensor(np.array([element for sublist in val_labels for element in sublist]).flatten()))

            model = DiagnosisModel(config, is_modal_1).to(config.device)
            if is_modal_1:
                modal_1_model = model
            else:
                modal_2_model = model

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)

            if is_modal_1:
                modal_1_val_loader = val_loader
            else:
                modal_2_val_loader = val_loader

            best_val_loss = float('inf')
            no_improvement = 0

            for epoch in range(config.max_epochs):
                model.train()
                epoch_loss = 0
                correct = 0

                for inputs, labels in train_loader:
                    inputs = inputs.to(config.device)
                    labels = labels.view(-1, 1).to(config.device)

                    optimizer.zero_grad()
                    # shape (4, 1),  (batch_size, 1)
                    outputs = model(inputs)
                    # outputs : common [batch_size,num_classes]
                    # here output : [batch_size, 1]

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    # _, predicted = torch.max(outputs.data, 1)

                    predicted = (outputs > curr_threshod).float()

                    correct += (predicted == labels).sum().item()

                train_loss = epoch_loss / len(train_loader)
                train_accuracy = 100 * correct / len(train_loader.dataset)

                print(f'{"Modal 1 " if is_modal_1 else 'Modal 2 ' }Epoch {epoch + 1}/{config.max_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%', end=',')

                # Each time verify and stop early
                model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(config.device)
                        labels = labels.view(-1, 1).to(config.device)

                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()

                        # _, predicted = torch.max(outputs.data, 1)

                        predicted = (outputs > curr_threshod).float()
                        val_correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = 100 * val_correct / len(val_loader.dataset)

                print(f' ValLoss: {val_loss:.4f}, ValAccuracy: {val_accuracy:.2f}%')

                # Record the loss and accuracy
                if is_modal_1 :
                    this_fold_train_losses_modal_1.append(train_loss)
                    this_fold_train_accuracies_modal_1.append(train_accuracy)
                    this_fold_val_losses_modal_1.append(val_loss)
                    this_fold_val_accuracies_modal_1.append(val_accuracy)
                else:
                    this_fold_train_losses_modal_2.append(train_loss)
                    this_fold_train_accuracies_modal_2.append(train_accuracy)
                    this_fold_val_losses_modal_2.append(val_loss)
                    this_fold_val_accuracies_modal_2.append(val_accuracy)

        all_train_accuracies_modal_1.append(this_fold_train_accuracies_modal_1)
        all_train_losses_modal_1.append(this_fold_train_losses_modal_1)
        all_val_accuracies_modal_1.append(this_fold_val_accuracies_modal_1)
        all_val_losses_modal_1.append(this_fold_val_losses_modal_1)

        all_train_accuracies_modal_2.append(this_fold_train_accuracies_modal_2)
        all_train_losses_modal_2.append(this_fold_train_losses_modal_2)
        all_val_accuracies_modal_2.append(this_fold_val_accuracies_modal_2)
        all_val_losses_modal_2.append(this_fold_val_losses_modal_2)

        # Record the prediction results of the validation set
        for model, which_modal in [(modal_1_model, "modal_1"), (modal_2_model, "modal_2")]:
            model.eval()

            with torch.no_grad():
                probs = []
                for inputs, labels in modal_1_val_loader if which_modal == "modal_1" else modal_2_val_loader:
                    # 【4,1】
                    outputs = model(inputs.to(config.device)).cpu().numpy()
                    probs.extend(outputs[:, 0])

            # Aggregate predicted probabilities by subject
            from collections import defaultdict
            subj_probs = defaultdict(list)
            for subj_id, p in zip(val_window_subject_map, probs): # val_window_subject_map is the subject id corresponding to each window
                subj_probs[subj_id].append(p)

            # save all prediction
            for subj_id in val_subjects_id:
                avg_prob = np.mean(subj_probs[subj_id])

                votes = np.array(subj_probs[subj_id]) > config.spiral_threshold if which_modal == "modal_1" else config.wavy_threshold
                final_pred = int(votes.mean() > config.spiral_vote_rate if which_modal == "modal_1" else config.wavy_vote_rate )

                if which_modal == 'modal_1':
                    results[subj_id]['modal1_predict'].append(final_pred)
                    results[subj_id]['modal1_votes'].append(votes)
                    results[subj_id]['modal1_raw_probs'].extend(subj_probs[subj_id])
                    results[subj_id]['modal1_probs'].append(avg_prob)
                else:
                    results[subj_id]['modal2_predict'].append(final_pred)
                    results[subj_id]['modal2_votes'].append(votes)
                    results[subj_id]['modal2_raw_probs'].extend(subj_probs[subj_id])
                    results[subj_id]['modal2_probs'].append(avg_prob)

    def to_csv(data):
        val = ""
        row = len(data)
        col = len(data[0])
        for line in data:
            for i in range(col-1):
                val += str(line[i]) + ","
            val += str(line[-1]) + '\n'
        return val

    with open(f'./final_acc_loss_{timestamp}.csv', 'w') as f:
        f.write("all train acc modal 1\n")
        f.write(to_csv(all_train_accuracies_modal_1))
        f.write("all train loss modal 1\n")
        f.write(to_csv(all_train_losses_modal_1))
        f.write("all val acc modal 1\n")
        f.write(to_csv(all_val_accuracies_modal_1))
        f.write("all val loss modal 1\n")
        f.write(to_csv(all_val_losses_modal_1))

        f.write("\nall train acc modal 2\n")
        f.write(to_csv(all_train_accuracies_modal_2))
        f.write("all train loss modal 2\n")
        f.write(to_csv(all_train_losses_modal_2))
        f.write("all val acc modal 2\n")
        f.write(to_csv(all_val_accuracies_modal_2))
        f.write("all val loss modal 2\n")
        f.write(to_csv(all_val_losses_modal_2))
        f.write("\n\n")

    # Calculate the final prediction result
    final_results = []
    for subj_id in valid_subjects_id:
        m1_prob = np.mean(results[subj_id]['modal1_probs'])
        m2_prob = np.mean(results[subj_id]['modal2_probs'])
        combined_prob = 0.5 * m1_prob + 0.5 * m2_prob

        final_results.append({
            'subject_id': subj_id,
            'modal1_prob': m1_prob,
            'modal1_pred': int(m1_prob > 0.5),
            'modal2_prob': m2_prob,
            'modal2_pred': int(m2_prob > 0.5),
            'combined_prob': combined_prob,
            'combined_pred': int(combined_prob > 0.5),
            'true_label': results[subj_id]['true_label']
        })

    def calculate_metrics(true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) != 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) != 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) != 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        }

    for model in ['modal1', 'modal2', 'combined']:
        y_true = [x['true_label'] for x in final_results]
        y_pred = [x[f'{model}_pred'] for x in final_results]
        metrics = calculate_metrics(y_true, y_pred)

        print(f"\n{model.upper()} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

    with open(f'./final_predictions_{timestamp}.csv', 'w') as f:
        for model in ['modal1', 'modal2', 'combined']:
            y_true = [x['true_label'] for x in final_results]
            y_pred = [x[f'{model}_pred'] for x in final_results]
            metrics = calculate_metrics(y_true, y_pred)

            f.write(f"\n{model.upper()} Performance:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall/Sensitivity: {metrics['recall']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1']:.4f}\n")

        f.write("subject_id,modal1_prob,modal1_pred,modal2_prob,modal2_pred,combined_prob,combined_pred,true_label\n")
        for res in final_results:
            line = f"{res['subject_id']},{res['modal1_prob']:.4f},{res['modal2_prob']:.4f},{res['combined_prob']:.4f}," \
                   f"{res['modal1_pred']},{res['modal2_pred']},{res['combined_pred']}," \
                   f"{res['true_label']}\n"
            f.write(line)

        for subj_id , statistics in results.items():
            f.write("\n")
            f.write(str(subj_id))
            f.write(f",{res['true_label']}\n")
            f.write(str(results[subj_id]['modal1_predict']))
            f.write("\n")
            f.write(str(results[subj_id]['modal1_votes']))
            f.write("\n")
            f.write(str(results[subj_id]['modal1_raw_probs']))
            f.write("\n")
            f.write(str(results[subj_id]['modal1_probs']))
        f.write("\n\n")
        for subj_id, statistics in results.items():
            f.write("\n")
            f.write(str(subj_id))
            f.write("\n")
            f.write(str(results[subj_id]['modal2_predict']))
            f.write("\n")
            f.write(str(results[subj_id]['modal2_votes']))
            f.write("\n")
            f.write(str(results[subj_id]['modal2_raw_probs']))
            f.write("\n")
            f.write(str(results[subj_id]['modal2_probs']))

    with open(f'./final_predictions_probs_{timestamp}.csv', 'w') as f:
        for subj_id , statistics in results.items():
            f.write(str(results[subj_id]['modal1_raw_probs']))
            f.write("\n")

        f.write("\n\n")
        for subj_id, statistics in results.items():
            f.write(str(results[subj_id]['modal2_raw_probs']))  # .append(subj_probs[subj_id])
            f.write("\n")

if __name__ == "__main__":
    main()