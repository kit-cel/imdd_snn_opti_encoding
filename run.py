'''
Define parameters, update and evaluate SNN and Gaussian RFE
'''

import torch 
import json  
import matplotlib.pyplot as plt
import functions.train 


#Set parameters                       
t_max = 10      #number of discrete time-steps SNN is run
n_enc = 10      # number of generated encoding signals 
path = 'results/tmax'+str(t_max)+'_nenc'+str(n_enc) + '/' 


# SNN and SNN update parameters (BPTT with SG)
bsz = int(1e5)       
var_t = -20          # training noise power [dB] 
epochs = int(4e4)
lr_snn = 1e-3
n_hid = 40              #number hidden neurons

# RFE update parameters (policy gradient)
bsz_t = 20                # number of variations of theta per loop (must be even number)
var_pol = 0.01
lr_pol = .5   

#Define values for evaluation
bsz_e = int(1e7)                #number of symbols for eval
var_e = torch.arange(-23,-14)   # noise variance for eval [dB]
loops = int(bsz_e/bsz)          #overall: bsz_e samples for evaluation

#IM/DD parameters LCD link
mod_order= 4              #4-PAM
baudrate = int(112e9)
usf = 3                 
dispersion = -5           # dispersion coefficient, in ps/nm/km
fiber_length= 4000        # fiber length [m]
wavelength = 1270         # wavelength of transmission [nm]
rrc_beta = 0.2
imdd_bias = 2.25
eq_tap = 7


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#Save config
config={
    'PAM-order': mod_order,
    'Signals per encoding': n_enc,
    'Equalizer taps':eq_tap,
    'Hidden neurons':n_hid,
    'SNN simulation steps':t_max,   
    'Learning rate BPTT':lr_snn,
    'Batchsize BPTT':bsz,       
    'Noise power training [dB]': var_t,       
    'Epochs':epochs,
    'Variance exploration noise':var_pol,
    'Learning rate policy':lr_pol,
    'Policy number of variations':bsz_t, 
    'Baudrate':baudrate,
    'Upsampling factor':usf,
    'Dispersion':dispersion,          
    'Fiber length':fiber_length,
    'Wavelength':wavelength,
    'Beta RRC':rrc_beta,
    'IM/DD bias':imdd_bias,
    }
    
with open(path+'config.json','w') as fp: 
        json.dump(config,fp)     #encode dict into json


# Init SNN and Encoder
SNN = functions.train.UPDATE(mod_order,n_enc,t_max,eq_tap,n_hid,bsz,var_t,epochs,lr_snn,bsz_t,var_pol,lr_pol,baudrate,usf,dispersion,fiber_length,wavelength,rrc_beta,imdd_bias,path,device)
SNN.train()
ser, ber, spikes = SNN.eval(var_e,loops)

#Save ser, ber and var_e 
torch.save(ser, path+'ser')
torch.save(ber, path+'ber')
torch.save(var_e, path+'var_e')
torch.save(spikes, path+'spikes')

plt.semilogy(var_e,ser,color='orange',linestyle='dashed',label='SER')
plt.semilogy(var_e,ber,color='orange',label='BER')
plt.xlabel('$\sigma^2$ (dB)')
plt.ylabel('SER')
plt.xlim(max(var_e),min(var_e))
plt.grid(True)
plt.legend()
plt.ylim(5e-6,3e-2)
plt.savefig(path+'SER_eval_surrogate.svg')
plt.close()

print('Done')

