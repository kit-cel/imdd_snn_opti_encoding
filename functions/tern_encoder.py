import torch
import pickle

import functions.bit as bit
import functions.quantizer as quantizer

class TERN_ENCODER:
    '''
    Ternary encoder for real-input only, x \in \mathbb{R}
    '''
    def __init__(self,neurons,t_max,device): 
        self.n = neurons
        self.t_max = t_max
        self.device =  device

    def encode(self,x):
        ''' Ternary encoding'''
        x = x-3         #remove mean value
        enc = self.encode_real(x.unsqueeze(dim=1)).unsqueeze(dim=0)
        enc = self.zero_pad(enc)        #Add zero-padding
        return enc

    def encode_real(self,x):
        d_x,bin_re = quantizer.midtread_binary_unipolar(torch.abs(x),w=self.n,x_max=4,device=self.device)
        sign = torch.permute(torch.sign(x).repeat(self.n,1,1),(1,2,0))
        y = bin_re*sign
        y = y.reshape( (y.shape[0],-1) )
        return y.to(self.device)
    
    def zero_pad(self, enc):
        enc = torch.cat((enc,torch.zeros(self.t_max-1,enc.size()[1],enc.size()[2]).to(self.device)),dim=0) #zero-padding for time behavior
        return enc

###################################################################################################
#                               Testing                                                           #                                                              
###################################################################################################
#
#if torch.cuda.is_available():
#    DEVICE = torch.device("cuda")
#else:
#    DEVICE = torch.device("cpu")
#print("Using " + str(DEVICE) + " for training.");
#
#w = 6
#
#numbers = torch.linspace(-3,3,257)
#print(numbers)
#print(numbers.shape)
##quantized = midrise(numbers,8,2);
##quantized = midtread_bipolar(numbers,8,2);
#quantized = midrise_bipolar(numbers,w,2);
#print(quantized)
#
##numbers = torch.linspace(0,255,256)
#numbers = torch.randn(64,1)+1
#numbers = torch.hstack( (torch.randn(64,1)+0,numbers) )
#numbers = torch.hstack( (torch.randn(64,1)-1,numbers) )
#numbers = torch.hstack( (torch.randn(64,1)+3,numbers) )
#
#Q = Lloyd_Max(w,DEVICE)
##Q = Lloyd_Max(w,DEVICE,torch.linspace(-2,2,2**w))
##Q = Lloyd_Max(w,DEVICE,torch.arange(0,2**w,1))
#print(Q.give_codebook())
#_,codebook = Q.optimize(numbers)
#print(Q.give_codebook())
#A_i,x_i = Q.quantize(numbers)
#
#print(x_i)
#print(A_i)
#
#self.__Q.load_codebook(name)
#delta_x = self.__Q.give_codebook()

