import torch

def pred_accuracy(z,y):
    compare = z.argmax(axis=1) == y.argmax(axis=1)
    return compare.float().mean()

def train_net(model, train_loader, test_data, epochs=200, learning_rate=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    train_history = []
    test_history = []
    for _ in range(epochs):
        cost = 0
        acc = 0
        batch_count = 0
        for inputs, targets in iter(train_loader):
            batchsize = inputs.shape[0]
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            acc += pred_accuracy(output, targets)
            cost += loss
            loss.backward()
            optimizer.step()
            batch_count += 1
        train_history.append((cost/batch_count,acc/batch_count,batchsize))
        cost = 0
        acc = 0
        batch_count = 0
        with torch.no_grad():
            output = model(test_data.x)
            cost = loss_fn(output, test_data.y)
            acc = pred_accuracy(output, test_data.y)
        test_history.append((cost,acc,batchsize))
    return train_history, test_history