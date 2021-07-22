import torch


def pred_accuracy(z, y):
    pred = (z > 0).float()
    compare = pred == y
    return compare.float().mean()


def train_net(
    model, train_loader, test_data, loss_fn=None, epochs=200, learning_rate=0.01
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
