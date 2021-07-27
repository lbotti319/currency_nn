import torch
from torch.utils.data import Dataset
from torch import nn


def pred_accuracy(z, y):
    pred = (z > 0).float()
    compare = pred == y
    return compare.float().mean()


def train_net(
    model, train_loader, test_data, loss_fn=None, epochs=200, learning_rate=0.01, device="cpu"
):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    train_history = []
    test_history = []
    for _ in range(epochs):
        cost = 0
        acc = 0
        batch_count = 0
        for inputs, targets in iter(train_loader):
            batchsize = inputs.shape[0]
            optimizer.zero_grad()
            output = model(inputs).squeeze(axis=1)
            loss = loss_fn(output, targets)
            cost += loss
            acc += pred_accuracy(output, targets)
            loss.backward()
            optimizer.step()
            batch_count += 1
        train_history.append((cost / batch_count, acc / batch_count, batchsize))
        cost = 0
        acc = 0
        batch_count = 0
        with torch.no_grad():
            output = model(test_data.x).squeeze()
            cost = loss_fn(output, test_data.y)
            acc = pred_accuracy(output, test_data.y)
        test_history.append((cost, acc, batchsize))
    return train_history, test_history

class BasicData(Dataset):
    def __init__(self, x, y, device='cpu'):
        super().__init__()
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, out_d):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
#         self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(num_layers * hidden_size, out_d)

    def forward(self, x):
        h_0 = x.new_zeros(
            self.num_layers, x.shape[0], self.hidden_size)

        c_0 = x.new_zeros(
            self.num_layers, x.shape[0], self.hidden_size)
#         print(type(x))
#         print(x.shape)
#         print(x)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
#         print(h_out.shape)

#         h_out = h_out.reshape(x.shape[0], self.hidden_size)
        h_out = h_out.transpose(1,0).reshape(x.shape[0],-1)
#         print(h_out.shape)

        out = self.fc(h_out)

        return out
