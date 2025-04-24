'Do receptive field encoding'

import torch
import matplotlib.pyplot as plt

  

class ARNOLD_ReceptiveFieldEncoder(torch.nn.Module):
    '''
    Neural encoding based on "Short-reach optical communications: a real-world task for neuromorphic hardware," by Arnold, Edelmann, von Bank, et al.
    Modified to fit update with policy gradient.
    '''
    def __init__(self, center: torch.Tensor, scaling: float, time_steps: float, dt: float, offset: float=0, cutoff: float = 0, device ='cuda'):
        super().__init__()
        self.device = device
        self.center = center.to(self.device)    # Peak of field
        self.n = len(center)
        self.scaling = torch.ones((self.n),device=self.device)*scaling            # slope of field
        self.offset = offset
        self.cutott = cutoff
        self.dt = dt
        self.time_steps = time_steps
        

    def encode(self, x: torch.Tensor,center,scaling) -> torch.Tensor:
        x = x.repeat(self.n,1)      #repeat data as many times as neurons
        t_spike = torch.zeros(x.size())
        for d1 in range(self.n):
            t_spike[d1,:] = (scaling[d1]*(x[d1,:]-center[d1]).abs() ).int()        #get spike times as integer values, int()=floor()
        t_spike = torch.clip(t_spike,min=0,max=self.time_steps-1).long()         #limit to min=0


        #Convert spike times into spike train
        enc = torch.zeros((self.time_steps,x.size()[1],self.n),device=self.device)       #DIM: time x bsz x neurons_in
        bsz_index = torch.linspace(0,x.size()[1]-1,x.size()[1]).long()
        for d1 in range(self.n):
            enc[t_spike[d1,:],bsz_index,d1] = 1
        return enc

class ARNOLD_ENC(torch.nn.Module):
    '''
    Neural encoding taken from "Short-reach optical communications: a real-world task for neuromorphic hardware," by Arnold, Edelmann, von Bank, et al.
    '''
    def __init__(self, scaling: float, offset: float, time_length: float,
                dt: float, center: torch.Tensor, cutoff: float = None):
        super().__init__()
        self.scaling = scaling
        self.offset = offset
        self.time_length = time_length
        self.dt = dt
        self.time_steps = int(time_length // dt) + 1
        self.center = center
        self.cutoff = cutoff if cutoff is not None else time_length

    def encode(self, trace: torch.Tensor, dummy1, dummy2) -> torch.Tensor:
        dev = trace.device

        # positive spike times
        times = self.scaling * torch.abs(trace.unsqueeze(-1) - self.center.to(dev)).reshape(trace.shape[0], -1)

        times[(times < 0) | (times > self.cutoff)] = self.time_length + self.dt
        times += self.offset

        bins = (times / self.dt + 1).long()
        mask = bins < self.time_steps
        mesh = torch.meshgrid([torch.arange(s) for s in times.shape], indexing="ij")

        indices = torch.stack(
            (bins.to(dev)[mask].reshape(-1),
            mesh[0].to(dev)[mask].reshape(-1),
            *(mesh[i].to(dev)[mask].reshape(-1)
            for i in range(1, len(mesh)))))

        spikes = torch.sparse_coo_tensor(
            indices, torch.ones(indices.shape[1]).to(dev),
            (self.time_steps, times.shape[0], *times.shape[1:]), dtype=int)

        return spikes.to_dense()