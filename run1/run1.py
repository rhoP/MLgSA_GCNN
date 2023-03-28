import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, DenseGCNConv, DenseGraphConv, global_mean_pool, ASAPooling, GATv2Conv, \
    GCNConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.CrossEntropyLoss()

train_loader = torch.load('../export/train_loader.pt')
test_loader = torch.load('../export/test_loader.pt')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 160)
        self.conv2 = GCNConv(160, 6)

    def forward(self, data, batch=None):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
train_log = []
test_log = []
acc = []


for epoch in range(1, 200 + 1):
    # Train Phase
    run_loss = 10000.
    model.train()
    for graph in train_loader:
        run_loss = 0.
        graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        y = graph.y.to(device)
        loss = loss_fn(out.view(-1), y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    train_log.append(run_loss)
    # Eval Phase
    model.eval()
    test_losses = []
    test_accu = []
    for graph in test_loader:
        graph.to(device)
        out = model(graph)
        y = graph.y.to(device)
        test_losses.append(loss_fn(out.view(-1), y).item())
        test_accu.append((out.argmax(dim=1) == y.argmax()).item())
    test_log.append(np.mean(test_losses))
    acc.append(np.sum(test_accu) / len(test_accu))
    show_info = '\nEpoch: {} -- Train: {}, Loss: {} Accuracy: {}'.format(epoch, train_log[-1], test_log[-1], acc[-1])
    print(show_info)


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "./states")


np.savetxt('train_log.npy', np.asarray(train_log))
np.savetxt('test_log.npy', np.asarray(test_log))
np.savetxt('acc.npy', np.asarray(acc))
