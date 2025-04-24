import torch
import torch.nn as nn

import norse.torch as norse
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell, LIFRecurrent, LIF
from norse.torch.module.leaky_integrator import LICell, LI
 
class SNN_MMSE(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, device, dt=1e-3):
        super(SNN_MMSE, self).__init__()
        self.device = device;
        self.dt = dt

        self.input_features  =  input_features
        self.hidden_features =  hidden_features
        self.output_features =  output_features

        self.p1 = norse.LIFParameters(
            tau_mem_inv=1/6e-3,
            tau_syn_inv=1/6e-3,
                    )
        self.p2 = norse.LIParameters(
            tau_mem_inv=1/6e-3,
            tau_syn_inv=1/6e-3,
                    )

        # SNN
        self.linear_1 = torch.nn.Linear(self.input_features, self.hidden_features, device=device, bias=None)
        self.lif = norse.LIFCell(p=self.p1,dt=self.dt)
        self.linear_2 = torch.nn.Linear(hidden_features, output_features, device=device, bias=None)
        self.li = norse.LICell(p=self.p2,dt=self.dt)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        T = input.shape[0]
        s_lif, s_li = None, None
        zs, ys, s_lifs, s_lis = [], [], [], []
        for ts in range(T):
            g1 = self.linear_1(input[ts])
            z, s_lif = self.lif(g1, s_lif)
            g2 = self.linear_2(z)
            y, s_li = self.li(g2, s_li)

            zs.append(z)
            ys.append(y)
            s_lifs.append(s_lif)
            s_lis.append(s_li)

        spikes = torch.stack(zs)
        traces = torch.stack(ys)

        score = torch.amax(traces, 0)
        spikes_mean = spikes.detach().sum(dim=2).sum(dim=0).mean()      

        return score, spikes_mean
    
