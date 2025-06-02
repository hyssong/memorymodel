import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class gru(nn.Module):
    def __init__(self, dim_input, lr=1e-3, dropout_rate=0):
        super(gru, self).__init__()

        self.dim_input = self.dim_output = dim_input
        self.dim_hidden = self.dim_memory = self.dim_input * 2

        self.i2h = nn.Linear(self.dim_input, self.dim_hidden * 3, bias=True)
        self.h2h = nn.Linear(self.dim_hidden, self.dim_hidden * 3, bias=True)
        self.h2o = nn.Linear(self.dim_hidden, self.dim_output, bias=True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.init_weight()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def init_weight(self):
        for name, parameter in self.named_parameters():
            if 'weight' in name and parameter.dim() >= 2:
                nn.init.orthogonal_(parameter)
            if 'bias' in name:
                nn.init.constant_(parameter, .1)
            if parameter.dim() == 0:
                nn.init.constant_(parameter, 0.5)

    def to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def to_numpy(self, data):
        return data.detach().cpu().numpy()

    def update_weights(self, loss):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def get_rand_states(self, scale=.1):
        return torch.randn(self.dim_hidden, ) * scale

    def forward(self, X):
        self.X = self.to_tensor(X)

        self.h_t = self.get_rand_states()
        self.m_t = self.get_rand_states()

        log_loss = torch.zeros((self.X.shape[1]-1,), dtype=torch.float32)
        log_acc = np.zeros((self.X.shape[1]-1,), dtype=np.float32)
        log_h = torch.zeros((self.X.shape[1]-1, self.dim_hidden), dtype=torch.float32)
        log_yhat = torch.zeros((self.X.shape[1]-1, self.dim_output), dtype=torch.float32)
        loss = 0
        for self.t in range(self.X.shape[1] - 1):
            self.x_t = self.X[:, self.t]
            self.y_t = self.X[:, self.t + 1]

            # input to hidden & hidden recurrence
            if self.dropout_rate>0:
                gate_x = self.dropout(self.i2h(self.x_t))
            else: gate_x = self.i2h(self.x_t)
            if self.dropout_rate>0:
                gate_h = self.dropout(self.h2h(self.h_t))
            else: gate_h = self.h2h(self.h_t)
            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()
            i_r, i_i, i_n = gate_x.chunk(3, 0)
            h_r, h_i, h_n = gate_h.chunk(3, 0)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + (resetgate * h_n))
            self.h_t = newgate + inputgate * (self.h_t - newgate)

            # prediction of the next time step
            if self.dropout_rate>0: self.yhat = self.dropout(self.h2o(self.h_t))
            else: self.yhat = self.h2o(self.h_t)

            # calculate loss & log
            loss_it = self.criterion(self.y_t, self.yhat)
            loss += loss_it
            log_loss[self.t] = loss_it.item()
            log_acc[self.t] = np.corrcoef(self.yhat.detach().numpy(), self.y_t)[0, 1]
            log_yhat[self.t] = self.yhat
            log_h[self.t] = self.h_t

        return loss, log_loss, log_acc, log_h, log_yhat

    @torch.no_grad()
    def forward_nograd(self, X):
        return self.forward(X)
