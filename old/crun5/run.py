import torch
import torch.nn as nn
from torch_geometric.nn import SSGConv, Sequential, TopKPooling, SAGPooling, global_mean_pool
import numpy as np
import torch
from torch.nn import Linear, ReLU, Dropout


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
            (Dropout(p=0.1), 'x -> x'),
            (SSGConv(in_channels, mp_units[0], alpha=0.3, K=20), 'x, edge_index, edge_weight -> x'),
            mp_act,
            (SAGPooling(mp_units[0], min_score=0.45), 'x, edge_index, edge_weight -> x')
        ]
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        mp1 = [
            (Dropout(p=0.5), 'x -> x'),
            (SSGConv(mp_units[0], mp_units[1], alpha=0.9, K=2), 'x, edge_index, edge_weight -> x'),
            mp_act,
            (SAGPooling(mp_units[1], min_score=0.3), 'x, edge_index, edge_weight -> x')
        ]
        self.mp1 = Sequential('x, edge_index, edge_weight', mp1)
        out_chan = mp_units[-1]

        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        self.mlp.append(mlp_act)

        # Final linear layers
        self.topk_pool = TopKPooling(mp_units[-1], ratio=30)
        self.comp = global_mean_pool

    def forward(self, data, batch=None):
        x = data.x[:, :4]
        edge_index = data.edge_index
        edge_weight = None
        # Propagate node feats
        x, c_edge_index, _, _, _, _ = self.mp(x, edge_index, edge_weight)
        x, c_edge_index, _, _, _, _ = self.mp1(x, c_edge_index, edge_weight=None)
        x = self.mlp(x)
        # x, edge_index, _, batch, _, _ = self.topk_pool(x, edge_index)

        # Cluster assignments (logits)
        s = self.comp(x, batch)

        return s


def main():
    # Data
    loss_fn = nn.CrossEntropyLoss()
    train_loader = torch.load('../../data/MLgSA/export/train_loader.pt')
    test_loader = torch.load('../../data/MLgSA/export/test_loader.pt')

    model = Net([1000, 1], "ELU", in_channels=4, n_clusters=2, mlp_units=[300, 2]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, weight_decay=1e-6)
    lmbda = lambda epoch: 0.9
    sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train_log = []
    test_log = []
    acc = []

    for epoch in range(1, 20 + 1):
        print("running job")
        # Train Phase
        run_loss = 0.
        iters = 0
        model.train()
        for graph in train_loader:
            graph.to(device)
            optimizer.zero_grad()
            out = model(graph)
            # print(out, graph.y)
            loss = loss_fn(out, graph.y)
            loss.backward()
            optimizer.step()
            run_loss += loss.detach().cpu()
            iters += 1
            print('\t\t loss: ', loss.item())
        train_log.append(run_loss/iters)
        # Eval Phase
        model.eval()
        test_losses = []
        test_accu = []
        for graph in test_loader:
            graph.to(device)
            out = model(graph)
            y = graph.y.to(device)
            test_losses.append(loss_fn(out, y).detach().cpu())
            test_accu.append((out.detach().cpu()).argmax(dim=1) == y.argmax())
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
