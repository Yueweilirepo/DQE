from typing import Dict
import torchinfo
import tqdm, math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ForecastDataset    

class LSTMModel(nn.Module):
    def __init__(self, window_size, feats, 
                 hidden_dim, pred_len, num_layers, batch_size, device) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.feats = feats
        self.device = device
        
        self.lstm_encoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, feats)
        
    def forward(self, src):
        _, decoder_hidden = self.lstm_encoder(src)
        cur_batch = src.shape[0]
        
        decoder_input = torch.zeros(cur_batch, 1, self.feats).to(self.device)
        outputs = torch.zeros(self.pred_len, cur_batch, self.feats).to(self.device)
        
        for t in range(self.pred_len):
            decoder_output, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)
            decoder_output = self.relu(decoder_output)
            decoder_input = self.fc(decoder_output)
            
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
            
        return outputs
    
class LSTMAD():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 hidden_dim=20,
                 num_layer=2,
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)

        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.feats = feats
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lr = lr
        self.validation_size = validation_size

        print('self.device: ', self.device)
        
        self.model = LSTMModel(self.window_size, feats, hidden_dim, self.pred_len, num_layer, batch_size=self.batch_size, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        # --- 修改开始 ---
        # 计算生成一个样本所需的最小长度
        min_len = self.window_size + self.pred_len

        # 原始切分点
        split_idx = int((1 - self.validation_size) * len(data))

        # 检查验证集是否足够长
        if len(data) - split_idx < min_len:
            print(
                f"警告: 验证集过小 ({len(data) - split_idx})，无法满足窗口大小 ({self.window_size}) + 预测长度 ({self.pred_len})。")
            # 强制调整 split_idx，给验证集留出刚好够用的数据（或者多留一点）
            # 这里我们留出 min_len + 5 个数据点以防万一，同时确保不让训练集变成负数
            ensure_valid_len = min_len + 2
            split_idx = len(data) - ensure_valid_len

            if split_idx < min_len:
                raise ValueError(f"数据总长度 ({len(data)}) 太短，无法同时满足训练集和验证集的窗口构建需求。")

            print(f"已动态调整切分点，验证集长度调整为: {len(data) - split_idx}")

        tsTrain = data[:split_idx]
        tsValid = data[split_idx:]
        # --- 修改结束 ---

        # tsTrain = data[:int((1-self.validation_size)*len(data))]
        # tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                # print('x: ', x.shape)       # (bs, win, feat)
                # print('target: ', target.shape)     # # (bs, pred_len, feat)

                self.optimizer.zero_grad()
                
                output = self.model(x)

                # print('output: ', output.shape)     # (pred_len, bs, feat)

                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x)
                    
                    output = output.view(-1, self.feats*self.pred_len)
                    target = target.view(-1, self.feats*self.pred_len)
                    
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
                    
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                    print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)

                mse = torch.sub(output, target).pow(2)
                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        # scores = 0.5 * (torch.log(self.sigma + self.eps) + (scores - self.mu)**2 / (self.sigma+self.eps))
        
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        y_hats = torch.cat(y_hats, dim=0)
        y_hats = y_hats.numpy()
        
        l, w = y_hats.shape
        
        # new_scores = np.zeros((l - self.pred_len, w))
        # for i in range(w):
        #     new_scores[:, i] = scores[self.pred_len - i:l-i, i]
        # scores = np.mean(new_scores, axis=1)
        # scores = np.pad(scores, (0, self.pred_len - 1), 'constant', constant_values=(0,0))
        
        # new_y_hats = np.zeros((l - self.pred_len, w))
        # for i in range(w):
        #     new_y_hats[:, i] = y_hats[self.pred_len - i:l-i, i]
        # y_hats = np.mean(new_y_hats, axis=1)
        # y_hats = np.pad(y_hats, (0, self.pred_len - 1), 'constant',constant_values=(0,0))

        assert scores.ndim == 1
        # self.y_hats = y_hats
        
        # print('scores: ', scores.shape)
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[: self.window_size+self.pred_len-1] = scores[0]
            padded_decision_scores_[self.window_size+self.pred_len-1 : ] = scores

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
