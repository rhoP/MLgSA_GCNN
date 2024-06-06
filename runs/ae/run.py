import torch
import torch.nn as nn
from torch_geometric.nn import SSGConv, Sequential, TopKPooling, global_mean_pool
import numpy as np
import torch
from torch.nn import Linear


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model
class Net(torch.nn.Module):
    def __init__(self,
                 mp_units,
                 mp_act,
                 in_channels,
                 n_clusters,
                 mlp_units=None,
                 mlp_act="Identity"):
        super().__init__()

        if mlp_units is None:
            mlp_units = []
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        # Message passing layers
        mp = [
            (SSGConv(in_channels, mp_units[0], alpha=0.3, K=20), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units) - 1):
            mp.append((SSGConv(in_channels, mp_units[0], alpha=0.3, K=20), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]

        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

        # Final linear layers
        self.topk_pool = TopKPooling(mp_units[-1], ratio=1000)
        self.comp = nn.Linear(out_chan, 1)

    def forward(self, data, batch=None):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)
        x, edge_index, _, batch, _, _ = self.topk_pool(x, edge_index)

        # Cluster assignments (logits)
        s = self.mlp(global_mean_pool(x, batch))

        return s


def main():
    # Data
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1037, 0.8962]))
    train_loader = torch.load('../../data/MLgSA/export/train_loader.pt')
    test_loader = torch.load('../../data/MLgSA/export/test_loader.pt')

    model = Net([20], "ELU", in_channels=4, n_clusters=2, mlp_units=[3000, 2]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4)
    lmbda = lambda epoch: 0.9
    sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train_log = []
    test_log = []
    acc = []

    for epoch in range(1, 18 + 1):
        print("running job")
        # Train Phase
        run_loss = 0.
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
            print('\t\t running loss: ', run_loss)
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
        sch.step()
        show_info = '\nEpoch: {} -- Train: {}, Loss: {} Accuracy: {}'.format(epoch, train_log[-1], test_log[-1],
                                                                             acc[-1])
        print(show_info)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./states")

    np.savetxt('train_log.npy', np.asarray(train_log))
    np.savetxt('test_log.npy', np.asarray(test_log))
    np.savetxt('acc.npy', np.asarray(acc))


if __name__ == '__main__':
    main()
