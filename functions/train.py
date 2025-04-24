''''
Script to init a SNN and linear RFE.
SNN is trained using BPTT with SG, RFE using policy gradients.
'''

import torch
import matplotlib.pyplot as plt
import functions.SNN 
import functions.im_dd 
import functions.MODEM 
import functions.rf_encoding


class UPDATE:

    def __init__(self,mod_order,n_enc,t_max,eq_tap,n_hid,bsz,var_t,epochs,lr_snn,bsz_t,var_pol,lr_pol,baudrate,usf,dispersion,fiber_length,wavelength,rrc_beta,imdd_bias,path,device):
        self.mod_order = mod_order 
        self.n_enc = n_enc        # neurons per encoding
        self.t_max = t_max          # number discrete simulation steps of SNN
        self.eq_tap = eq_tap      # number samples used for equalization
        self.n_in = n_enc*eq_tap
        self.n_hid = n_hid          # number of hidden layer LIF-Recurrent-neurons
        self.bsz = bsz              #batchsize of BPTT with SG
        self.var_t = var_t          # noise variance at which SNN is optimized (dB)
        self.epochs = epochs        # number of epochs
        self.lr_snn = lr_snn        #learning rate BPTT
        self.dt = 5e-4

        self.bsz_t = bsz_t          # number parameter-variation per update step
        self.var_pol = var_pol          # variance of gaussian policy
        self.lr_pol = lr_pol         # learning rate 
        self.path = path            # path where to store results
        self.device = device        # device to use
        self.lossfn = torch.nn.CrossEntropyLoss()       #loss function for snn optimization and metric for policy-gradient update

        #parameter of IM/DD
        self.baudrate = baudrate 
        self.usf = usf
        self.dispersion =dispersion
        self.fiber_length = fiber_length
        self.wavelength = wavelength
        self.rrc_beta = rrc_beta
        self.imdd_bias = imdd_bias


        #Define network architecture, channel, transmitter and encoding
        self.model = functions.SNN.SNN_MMSE(input_features= self.n_in,
                                            hidden_features= self.n_hid,
                                            output_features= self.mod_order,
                                            dt = self.dt,
                                            device=self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.lr_snn)

        self.channel = functions.im_dd.IM_DD_CHANNEL(SYMCNT=self.bsz,
                                               Baud_Rate =  self.baudrate,
                                               rate = int(torch.log2(torch.tensor(self.mod_order))),
                                               USF = self.usf,
                                               D = self.dispersion,
                                               channel_length = self.fiber_length,
                                               wavelen = self.wavelength,
                                               RRC_beta = self.rrc_beta,
                                               bias = self.imdd_bias,
                                               device=self.device)
        self.modem = functions.MODEM.MODEM(modulation = 'PAM',
                                           m = int(torch.log2(torch.tensor(mod_order))),
                                           gray = 'yes',
                                           device = self.device,
                                           IMDD = False)


        self.encoder = functions.rf_encoding.ARNOLD_ReceptiveFieldEncoder(center = torch.linspace(0,7,self.n_enc),
                                                                        scaling = 7*self.t_max/6,
                                                                        dt = self.dt,
                                                                        time_steps = self.t_max      
        )

        #Init variables for best encoding parameters and best loss seen so far
        self.best_loss = torch.tensor(10,device=self.device)
        self.best_center = None 
        self.best_scaling = None
               

        

    def train(self):       
        losses = []
        ser = []
        for e in range(self.epochs):
            loss_batch,ser_batch = self.snn_update()
            losses.append(loss_batch) 
            ser.append(ser_batch)
            if e<=10000:        #update encoding during first 10.000 epochs
                self.enc_update(loss_batch)

            if e%1000 == 0:     #print progress from time to time
                print('Epoch:',e,'       Loss:',loss_batch,'            SER',ser_batch)


        #Save model and encoding parameters 
        torch.save(self.model.state_dict(),self.path+'SNN_param')
        torch.save(self.encoder.center,self.path+'rfe_mean')
        torch.save(self.encoder.scaling,self.path+'rfe_std')


        #Plot losses
        plt.semilogy(torch.tensor(losses).cpu(),color='b',label='CE Loss')
        plt.semilogy(torch.tensor(ser).cpu(),color='orange',label='SER')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('CE Loss')
        plt.grid(True)
        plt.ylim(1e-3,1)
        plt.savefig(self.path+'loss.svg')
        plt.close()
        print('Training done')
        return 
    
    def snn_update(self):
        ''' Update SNN using BPTT with SG
        INPUT: None
        OUTPUT: loss and SER of update 
        '''
        rx, labels, _ = self.create_data(self.bsz,self.var_t)
        enc = self.encode(rx,self.encoder.center,self.encoder.scaling)
        
        out,_ = self.model.forward(enc.float())
        loss = self.lossfn(out,labels.long())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ser = torch.ne(out.argmax(dim=1),labels).sum()/self.bsz

        return loss.detach(), ser.detach()
        

    def enc_update(self,loss_batch):
        '''Update the parameters of the neural encoding using policy gradient
        INPUT: most recent loss after snn-update
        OUPUT: None
        '''
  
        #Draw variations from encoder initialization
        noise_center = (torch.randn((self.n_enc,self.bsz_t))*self.var_pol).to(self.device)
        noise_scaling = (torch.randn((self.n_enc,self.bsz_t))*self.var_pol).to(self.device)
        center_var = self.encoder.center.unsqueeze(dim=1).repeat(1,self.bsz_t) + noise_center
        scaling_var = self.encoder.scaling.unsqueeze(dim=1).repeat(1,self.bsz_t) + noise_scaling

        #Evaluate each variation
        loss_var = torch.zeros((self.bsz_t),device=self.device)
        for d1 in range(self.bsz_t):    
            rx, labels, _ = self.create_data(self.bsz,self.var_t)       
            enc = self.encode(rx,center_var[:,d1],scaling_var[:,d1])        
            out,_ = self.model.forward(enc) # .sum(dim=0)
            loss_var[d1] = self.lossfn(out,labels.long()).detach()

        #Check if theres new best parameter set
        if loss_var.min() <= self.best_loss:
            index = loss_var.argmin() 
            self.best_loss = loss_var[index] 
            self.best_center = center_var[:,index]
            self.best_scaling = scaling_var[:,index]

        #Update step using SGD 
        scaling = self.lr_pol*((loss_var-loss_batch)/loss_batch).unsqueeze(dim=0).repeat(self.n_enc,1)
        self.encoder.center = self.encoder.center + (scaling * noise_center).sum(dim=1)
        self.encoder.scaling = self.encoder.scaling + (scaling * noise_scaling).sum(dim=1)
        self.encoder.center = (self.encoder.center+self.best_center)/2
        self.encoder.scaling = (self.encoder.scaling+self.best_scaling)/2
       
        return
    
    def create_data(self,bsz,var_n):
        '''
        Generate bsz random symbols and simulate IM/DD-link transmission;
        Channel output is ternary encoded and prepared as SNN-input.
        INPUT: Number of samples to be generated, noise power of AWGN
        OUTPUT: rx-data, labels of tx-data, bits of tx-data
        '''
        #Create random data and apply channel
        bits = torch.randint(0,2,size=(int(bsz*torch.log2(torch.tensor(self.mod_order))),),device=self.device)
        labels, tx = self.modem.modulate(bits)          #Bits -> Symbols and labels
        rx = self.channel.apply(tx*torch.sqrt(torch.tensor(5)),var_n)   #Get TX \in {-3,-1,1,3}
        return rx, labels, bits
        
    def encode(self,x,center,scaling):
        '''Encode and reshape into correct format 
        INPUT: data to be encoded, center and scaling of neural encoder
        OUTPUT: encoded data
        '''
        #Apply encoding to rx 
        x_enc = self.encoder.encode(x,center,scaling)             # encode each symbol

        enc= x_enc.clone()                        # Initialize input to enable cat
        for d1 in range(-int((self.eq_tap-1)/2),int((self.eq_tap-1)/2)+1,1):
            enc = torch.cat((enc,torch.roll(x_enc,shifts=d1,dims=1)),dim=2)
        enc = enc[:,:,self.n_enc:]         #Discard initialized x of input
        
        return enc

    
    
    def eval(self,var_e,loops):
        ser = torch.zeros((len(var_e),loops))
        ber = torch.zeros((len(var_e),loops))
        spikes = torch.zeros((len(var_e),loops))
        for d1 in range(len(var_e)):
            for d2 in range(loops):         #loop to allow for more samples during evaluation
                rx,label,bit = self.create_data(self.bsz,var_e[d1])
                enc = self.encode(rx,self.encoder.center,self.encoder.scaling)          
                out,spikes[d1,d2] = self.model.forward(enc.float())


                #Calculate SER
                labels_est = out.argmax(dim=1)
                ser[d1,d2] = torch.ne(label,labels_est).sum()/self.bsz

                #Calculate BER
                _,bits_est = self.modem._MODEM__PAM_A_to_bits(labels_est) 
                ber[d1,d2] = torch.ne(bit,bits_est).sum()/len(bit)
                
            print('%i db done'%var_e[d1])    
        return ser.mean(dim=1),ber.mean(dim=1), spikes.mean(dim=1)
