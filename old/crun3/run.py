import torch
import torch.nn as nn
from torch_geometric.nn import SSGConv, Sequential, TopKPooling, global_max_pool, global_mean_pool
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
        self.mp = [
            SSGConv(in_channels, mp_units[0], alpha=0.7, K=20),
            mp_act
        ]
        for i in range(len(mp_units) - 1):
            self.mp.append(SSGConv(mp_units[i], mp_units[i + 1], alpha=0.7/(i+1), K=20))
            self.mp.append(mp_act)
        # self.mp = Sequential('x, edge_index, edge_weight', mp)
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
        self.comp = global_max_pool

    def forward(self, data, batch=None):
        x = data.x[:, :4]
        edge_index = data.edge_index
        edge_weight = None
        # Propagate node feats
        x = self.mp[0](x, edge_index, edge_weight)
        x = self.mp[1](x)
        x = self.mp[2](x, edge_index, edge_weight)
        x = self.mp[3](x)
        x = self.mp[5](self.mp[4](x, edge_index, edge_weight)) + x
        x = self.mp[7](self.mp[6](x, edge_index, edge_weight)) + x
        x = self.mp[9](self.mp[8](x, edge_index, edge_weight)) + x
        # min_value, min_index = torch.min(x4, dim=0, keepdim=False)
        # print("x5", x5.shape)
        # print("min val ", min_value, "min ind", min_index)
        x = self.mlp(x)
        x, edge_index, _, batch, _, _ = self.topk_pool(x, edge_index)
        # print("s", s1.shape)
        # Cluster assignments (logits)
        x = self.comp(x, batch)

        return x

def main():
    # Data
    loss_fn = nn.CrossEntropyLoss()
    train_loader = torch.load('../../data/MLgSA/export/train_loader.pt')
    test_loader = torch.load('../../data/MLgSA/export/test_loader.pt')

    model = Net([1000, 1000, 1000, 1000, 1], "ELU", in_channels=4, n_clusters=2, mlp_units=[300, 2]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, weight_decay=1e-6)
    lmbda = lambda epoch: 0.9
    sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train_log = []
    test_log = []
    acc = []

    for epoch in range(1, 100 + 1):
        print("running job")
        # Train Phase
        run_loss = 0.
        iters = 0
        model.train()
        for graph in train_loader:
            print("0")
            graph.to(device)
            optimizer.zero_grad()
            out = model(graph)
            # print(out, graph.y)
            loss = loss_fn(out, graph.y)
            print("1")
            loss.backward()
            optimizer.step()
            print("2")
            run_loss += loss.detach().cpu()
            iters += 1
            print('\t\t almost: ')
        train_log.append(run_loss/iters)
        # Eval Phase
        model.eval()
        test_losses = []
        test_accu = []
        for graph in test_loader:
            graph.to(device)
            out = model(graph)
            y = graph.y.to(device)
            test_losses.append(loss_fn(out, y).item())
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
