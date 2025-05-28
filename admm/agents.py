import numpy as np
import torch.nn as nn
import torch
from admm.utils import *
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

class AgentBase:
    
        def __init__(self, loss: nn.Module, model: nn.Module, train_loader: DataLoader, 
                    epochs: int, device: str, lr: float) -> None:        
            
            self.device = device
            self.model = model.to(device)
            self.lr = lr
            self.epochs = epochs
            self.criterion = loss
            self.train_loader = train_loader

        def copy_params(self, params): #  FedLearn, FedADMM, Scaffold_Agent, TrainingAgentADMM
            copy = [torch.zeros(param.shape).to(self.device).copy_(param) for param in params]
            return copy
        
        def set_parameters(self, parameters, model: nn.Module) -> nn.Module:  
            # FedLearn, FedADMM, Scaffold_Agent, TrainingAgentADMM-(no if, only else)
            """Change the parameters of the model using the given ones."""

            params_dict = zip(model.state_dict().keys(), parameters)
              
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            
            model_dict = model.state_dict()
            new_state_dict={k:v if v.size()==model_dict[k].size()  else  model_dict[k] for k,v in zip(model_dict.keys(), state_dict.values())}

            self.model.load_state_dict(new_state_dict, strict=False)
            return self.model
        
        def get_parameters(self, model): 
            """Return the parameters of the current net."""
            return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FedLearn(AgentBase):

    """
    Distributed event-based ADMM for federated learning
    """
    
    def __init__(self, loss: nn.Module, model: nn.Module, train_loader: DataLoader, 
                 rho: float, epochs: int, device: str, lr: float) -> None:        
        super().__init__(loss, model, train_loader, epochs, device, lr)
        self.rho = rho
        self.num_samples = len(train_loader.dataset)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        
    def primal_update(self, global_params) -> None:

        self.model = self.set_parameters(parameters=global_params, model=self.model)     
        global_copy = self.copy_params(self.model.parameters())

        # Solve argmin problem
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                if self.rho > 0:
                    for param, global_param in zip(self.model.parameters(), global_copy):
                        prox += torch.norm(param - global_param.data, p='fro')**2
                 
                output = self.model(data)
                   
                loss = self.criterion(output, target) + prox*self.rho/2
                if torch.isnan(loss):
                    print(f"NaN loss encountered at batch {i}")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping

                self.optimizer.step() 

 
class FedADMM(AgentBase):

    def __init__(self, rho: int, N: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float) -> None: 
        super().__init__(loss, model, train_loader, epochs, device, lr)
        self.primal_avg = None
        self.rho=rho
        self.N=N
        self.broadcast = False
        self.last_communicated = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        try:
            net = self.model.network
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.data_ratio = data_ratio

    def update(self, global_params) -> None:
        # Solve argmin problem
        self.primal_avg = self.copy_params(global_params)
        self.dual_update()
        using_global=True
        if using_global: self.model = self.set_parameters(parameters=global_params, model=self.model)
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                    prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
                output = self.model(data)
            
                loss = self.criterion(output, target) + prox*self.rho/2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        
        self.update_residual()

    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        add_params(self.lam, primal_copy)
    
    def update_residual(self):
        # Current local z-value
        self.residual = self.copy_params(self.model.parameters())
        add_params(self.residual, self.lam)
        # scale_params(self.residual, a=1/self.N)

    
class Scaffold_Agent(AgentBase):

    """ Primal Update is modified for scaffold    """
    
    def __init__(self, loss: nn.Module, model: nn.Module, train_loader: DataLoader, 
                 epochs: int, device: str, lr: float) -> None:        
        super().__init__(loss, model, train_loader, epochs, device, lr)
        self.num_samples = len(train_loader.dataset)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.c_i = [torch.zeros(param.shape).to(self.device) for param in model.parameters()]


    def primal_update(self, global_params, global_c) -> None:
       
        #self.model = self.set_parameters(global_params, model=self.model)   
        self.model = self.set_parameters(parameters=global_params, model=self.model)     
        for _ in range(self.epochs):
            # compute mini batch gradient g_i(y_i)                
            # add the scaffold correction to the gradient g_i(y_i) - c_i + c
            # update the model
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                # compute the loss
                output=self.model(data)
                
                loss = self.criterion(output, target)
                # Get the gradients form loss.backward()
                self.optimizer.zero_grad()
                loss.backward()
                # Get the gradients
                grad = [param.grad for param in self.model.parameters()]
                # Add the scaffold correction to the gradient g_i(y_i) - c_i + c
                for i, (g, c_i, c) in enumerate(zip(grad, self.c_i, global_c)):
                    g -= c_i - c
                    grad[i] = g
                
                for param, gr in zip(self.model.parameters(), grad):
                    param.grad = gr
                self.optimizer.step()


        # Extract the parameter change from the model
        # delta_y_i = y_i-x
        delta_y_i = [ param - global_param for param, global_param in zip(self.model.parameters(), global_params)]
        # norm of delta_y_i
        # norm_delta_y_i = sum([torch.norm(param, p='fro').item()**2 for param in delta_y_i])
        # compare the norm of delta_y_i with delta_prime
        #print(f"norm_delta_y_i: {norm_delta_y_i}")
        self.delta_y_i = delta_y_i
        #print("Primal Update done.")
        #print("Corrector Update starts.")
        self.c_i_update(global_params, global_c)
        #print("Corrector Update done.")

    
                
    def c_i_update(self, global_params, global_c) -> None: 
        #print("Corrector Update")
        K = self.epochs # number of local epochs
        n_l = self.lr # local learning rate	
        # c_i = c_i - c + 1/K n_l (x-y_i)
        c_i_copy = self.copy_params(self.c_i)
        c_i_old = self.copy_params(self.c_i)
        subtract_params(c_i_copy, global_c) 
        delta_y_i_copy = self.copy_params(self.delta_y_i)
        scale_params(delta_y_i_copy, a=1/K*n_l)
        subtract_params(c_i_copy, delta_y_i_copy)
        self.c_i = c_i_copy
        # delta_c_i = - c + 1/K n_l (x-y_i)
        # delta_c_i = c_i - c_i_old
        subtract_params(c_i_copy, c_i_old)
        self.delta_c_i = c_i_copy      



class TrainingAgentADMM(AgentBase):

    """
    Distributed event-based ADMM for federated learning
    """
    
    def __init__(self, rho: int, N: int, delta: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float, global_weight: float) -> None:        
        super().__init__(loss, model, train_loader, epochs, device, lr)
        self.primal_avg = None
        self.rho=rho
        self.N=N
        self.delta = delta
        self.broadcast = False
        self.global_weight = global_weight
        self.last_communicated = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()] # dual variable
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.data_ratio = data_ratio
        # Get number of params in model
        self.total_params = sum(param.numel() for param in self.model.parameters())
        # Set up learning rate scheduler # We do not want this!
        # self.stepper = StepLR(optimizer=self.optimizer, gamma=0.95, step_size=1)

    def primal_update(self,round,params) -> None:
        self.primal_avg = self.copy_params(params)
        if round > 0: self.dual_update()
        using_global = 1
        # Solve argmin problem
        if using_global == 1: self.model = self.set_parameters(parameters=params, model=self.model)
        # Solve argmin problem
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                    prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    pred = self.model(data)          
                    loss = self.criterion(pred, target) + prox*self.rho/2 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
   
        # check for how much paramters changed
        change = 0
        for old_residual, updated_param, dual_param in zip(self.last_communicated, self.model.parameters(), self.lam):
            with torch.no_grad():
                change += torch.norm(old_residual.data-updated_param.data-dual_param.data,p='fro').item()**2 
                # ||x^i_{k+1}+u^i_{k+1}-x^i_{[k]}+u^i_{[k]}||^2, fro: squared frobenius norm
        d_i = np.sqrt(change)

        # If "send on delta" then update residual and broadcast to other agents
        if d_i >= self.delta:      
            self.update_residual()
            self.last_communicated = self.copy_params(self.model.parameters())
            add_params(self.last_communicated, self.lam)
            self.broadcast = True
        else:
            self.broadcast = False

        return d_i
    
    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        add_params(self.lam, primal_copy)

    def update_residual(self):
        # Current local z-value
        self.residual = self.copy_params(self.model.parameters())
        add_params(self.residual, self.lam) # x^i_{k+1}+u^i_{k+1}
        subtract_params(self.residual, self.last_communicated) # x^i_{k+1}+u^i_{k+1}-x^i_{[k]}+u^i_{[k]}


