import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, dense_diff_pool, GCNConv, DenseSAGEConv, global_mean_pool
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, remove_self_loops, subgraph, k_hop_subgraph
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on: ', device)


################################ Model

class BasicMessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicMessagePassing, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))


###############################################

class GNNWithDenseDiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(GNNWithDenseDiffPool, self).__init__()
        self.conv1 = BasicMessagePassing(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, 125)
        self.lin2 = nn.Linear(125, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, 2)
        
    def forward(self, data, batch=None):
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        x = self.conv1(x, edge_index)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x1 = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x1) + x)
        x = F.relu(self.lin3(x))
        out = global_mean_pool(x, batch)
        return out
##################################################################



def main():
    # Data
    
    train_loader = torch.load(r'C:\Users\Rohit\Documents\Projects\data\MLgSA\train_loader.pt')
    test_loader = torch.load(r'C:\Users\Rohit\Documents\Projects\data\MLgSA\test_loader.pt')

    in_channels = 1
    hidden_channels = 4
    out_channels = 2
    num_clusters = 3  # Number of clusters for DiffPool

    model = GNNWithDenseDiffPool(in_channels, hidden_channels, out_channels, num_clusters).to(device)
    loss_fn = nn.CrossEntropyLoss()

    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
    lmbda = lambda epoch: 0.9
    sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train_log = []
    test_log = []
    acc = []

    for epoch in range(1, 1000 + 1):
        print("running job")
        # Train Phase
        run_loss = 0.
        iters = 0
        model.train()
        for graph in train_loader:
            graph.to(device)
            optimizer.zero_grad()
            out = model(graph)
            loss = loss_fn(out, graph.y)
            loss.backward()
            #for name, param in model.named_parameters():
            #    print(name, param.grad)
            optimizer.step()
            
            run_loss += loss.detach().cpu()
            iters += 1
        train_log.append(run_loss/iters)


        # Eval Phase
        model.eval()
        test_losses = []
        test_accu = []
        for graph in test_loader:
            graph.to(device)
            out = model(graph)
            test_losses.append(loss_fn(out, graph.y).item())
            test_accu.append(out.argmax(dim=1)==graph.y)
        test_log.append(np.mean(test_losses))
        acc.append(np.sum(test_accu) / len(test_accu))
        show_info = '\nEpoch: {} -- Train: {}, Loss: {}, Accuracy: {}'.format(epoch, train_log[-1], test_log[-1], acc[-1])
        print(show_info)

    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./states")

    np.savetxt('train_log.npy', np.asarray(train_log, dtype=object))
    np.savetxt('test_log.npy', np.asarray(test_log, dtype=object))
    np.savetxt('acc.npy', np.asarray(acc, dtype=object))


if __name__ == '__main__':
    main()
