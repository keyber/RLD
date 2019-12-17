import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import torch
from torch import nn


class Anim:
    def __init__(self, action_space, nb_frames):
        self.nb_frames = nb_frames
        
        plt.subplot(1, 2, 1)
        plt.title("traj & V")
        plt.xlim(-1, 1)
        plt.ylim(-10, 50)
        # plt.yticks()
        # plt.semilogy()
        self.y_true = [plt.plot([],[],'.', color=matplotlib.colors.hsv_to_rgb((epoch / nb_frames, 1., 1.)))[0]
                       for epoch in range(nb_frames)]
        self.y_pred = plt.plot([],[],'.', color=(0,0,0,.5))[0]
        
        plt.subplot(1, 2, 2)
        plt.title("Q")
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        self.action_q = [plt.plot([], [], '.', color="black")[0] for _action in range(action_space)]
        
        self._print_process = None
        

    def log(self, trajectories_dict, V, Q):
        def cat(x_, y_, axis=0):
            if x_ is None:
                return y_
            return np.concatenate((x_, y_), axis=axis)
        
        last_epoch = max(trajectories_dict.keys())
        x, state_v, action_q = None, None, None
        
        for epoch, trajectories in trajectories_dict.items():
            i, s, _a, r = next(iter(trajectories))
            s_1 = s[:, 2] - s[:, 0]
            x = cat(x, s_1.reshape(-1).numpy())
            state_v = cat(state_v, V(s).reshape(-1).detach())
            action_q = cat(action_q, Q(s).detach().transpose(0,1), axis=1)
            
            if epoch == last_epoch:
                self.y_true[epoch].set_data(s_1.reshape(-1), r.reshape(-1))    
        
        self.y_pred.set_data(x, state_v)
        for action in range(len(self.action_q)-1):
            self.action_q[action].set_data(x, action_q[action])        
        
        # affichera le plot sans bloquer l'entraînemnt pendant env.render()
        plt.pause(.1)
    
    """
    import multiprocessing
    pb serveur X
    def show(self, non_blocking=True):
        if non_blocking:
            if self._print_process is not None:
                self._print_process.terminate()
            
            self._print_process = multiprocessing.Process(target=plt.show)
            self._print_process.start()
        else:
            plt.show()
    """
    def write_anim(self, path):
        pass


class NN_Q(nn.Module):
    def __init__(self, inSize, outSize, layers, eps=1e-3):
        super().__init__()
        self.eps = eps
        
        l = []
        for x in layers:
            l.append(nn.Linear(inSize, x))
            l.append(nn.LeakyReLU())
            inSize = x
        
        l.append(nn.Linear(inSize, outSize))
        l.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*l)
    
    def forward(self, x: torch.Tensor):
        assert x.ndimension() == 2
        
        score = self.layers[:-1](x)
        assert_numeric(score)
        
        proba = self.layers[-1](score)
        assert_numeric(proba)
        
        zeroes = proba == 0
        if torch.any(zeroes):
            n_zero = torch.sum(zeroes, dim=1)
            proba = (proba.transpose(0, 1) * (- n_zero * self.eps + 1)).transpose(0, 1)
            proba[zeroes] += self.eps
        
        assert torch.all(torch.abs(torch.sum(proba, dim=1) - 1.) < 1e-4), (torch.sum(proba, dim=1))
        assert_numeric(proba)
        return proba


class NN_V(nn.Module):
    def __init__(self, inSize, layers):
        super().__init__()
        
        l = []
        for x in layers:
            l.append(nn.Linear(inSize, x))
            l.append(nn.LeakyReLU())
            inSize = x
        
        l.append(nn.Linear(inSize, 1))
        
        self.layers = nn.Sequential(*l)
    
    def forward(self, x):
        # assert x.ndimension()==2
        res = self.layers(x).squeeze(1)
        assert_numeric(res)
        return res

def show_trajectory(Q, V, traj):
    for s in traj:
        v, q = get_values(Q, V, s)
        print(s[2] - s[0], "%.2f" % v.item(), q.data)
    print()

def get_values(Q, V, state):
    if type(state) != torch.Tensor:
        state = torch.tensor(state, dtype=torch.float32)
    state = state.unsqueeze(0)

    return V(state), Q(state)

def pprint(traj, V):
    print("ind \tstate \t rew \t v")
    for i, s, _a, r in traj:
        v = V(s)
        for ii, ss, rr, vv in zip(i, s, r, v):
            print("%d   \t%.2f\t%.2f\t%.2f" % (ii.item(), (ss[2] - ss[0]).item(), rr.item(), vv.item()))
            # print("%d   \t%.2f\t%.2f\t%.2f"%(ii.item(), ss.item(), rr.item(), vv.item()))


def assert_numeric(*A):
    for a in A:
        if isinstance(a, torch.nn.Module):
            for b in a.parameters():
                assert_numeric(b.data)
        
        elif type(a) is torch.Tensor:
            assert not torch.any(torch.isnan(a) | torch.isinf(a))
        
        else:
            for b in a:
                assert_numeric(b)
"""
def plot_v(self, n=10):
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    X = [np.linspace(x_min, x_max, num=n) for (x_min, x_max) in self.space_bounds]
    X = np.meshgrid(*X)
    X = torch.tensor(X, dtype=torch.float32)
    assert X.shape == (self.state_space, *([n] * self.state_space))

    x = X[:, 2] - X[:, 0]

    X = X.reshape((self.state_space, -1)).transpose(0, 1)
    z = self.V(X).detach()
    print(z.shape)

    x = X[:, 2] - X[:, 0]
    y = X[:, 3] - X[:, 1]
    print(x.shape)
    surf = ax.plot_surface(x, y, z, cmap=cm.get_cmap("coolwarm"),
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
"""