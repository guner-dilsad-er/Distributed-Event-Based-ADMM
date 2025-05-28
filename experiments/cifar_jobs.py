from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from admm.models import Cifar10CNN, FCNet

from admm.agents import FedLearn, FedADMM, Scaffold_Agent, TrainingAgentADMM
from admm.servers import FedAgg, EventADMM, InexactADMM, DriftServer
from admm.utils import average_params
from typing import List
import numpy as np

class FedADMMJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, num_agents: int, epochs: int, rate: float,
                cifar: bool, mnist: bool, seed: int):
        self.cifar = cifar
        self.mnist = mnist
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.epochs = epochs
        self.rho = 0.01
        self.rate = rate
        self.device = device
        self.num_agents = num_agents
        self.seed = seed

    def run(self):
        rates=[self.rate]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []
        total_samples = sum([len(loader.dataset) for loader in self.train_loaders])

        if self.cifar: data_dir = 'figure_data/cifar/FedADMM/'
        if self.mnist: data_dir = 'figure_data/mnist/FedADMM/'

        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(self.seed)
                if self.cifar: 
                    model = Cifar10CNN()
                elif self.mnist: 
                    model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

                self.agents.append(
                    FedADMM(
                        rho=self.rho,
                        N=self.num_agents,
                        model=model,
                        loss=nn.CrossEntropyLoss(),
                        train_loader=loader,
                        epochs=self.epochs,
                        device=self.device,
                        lr=self.lr,
                        data_ratio=len(loader.dataset)/total_samples
                    )
                )

            torch.manual_seed(self.seed)
            if self.cifar: 
                global_model = Cifar10CNN()

            elif self.mnist: 
                global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

            k0 = 1
            server = InexactADMM(clients=self.agents, C=rate, t_max=self.t_max,
                                 model=global_model, device=self.device, num_clients=self.num_agents, k0=k0)
            server.spin(loader=self.val_loader)

            self.acc_per_rate[i,:] = server.val_accs
            self.rate_per_rate[i,:] = server.rates
            load = sum(self.rate_per_rate[i,:])/self.t_max
            acc, _ = server.validate_global(loader=self.test_loader)
            print(f'Test accuracy for rate {rate} = {acc}')
            self.loads.append(load)
            self.test_accs.append(acc.cpu().numpy())

            np.save(file=data_dir+'accs_'+str(rate*100), arr=self.acc_per_rate)
            np.save(file=data_dir+'loads_'+str(rate*100), arr=self.loads)
        
        print(f'saved: {self.acc_per_rate} and {self.loads}')




class SCAFFOLDJob:
    "Job Class for the SCAFFOLD"
    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, epochs: int, rate: float,  num_agents: int,
                cifar: bool, mnist: bool,  seed: int):
        
        self.cifar = cifar
        self.mnist = mnist
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.t_max = t_max
        self.lr = lr
        self.rate = rate
        self.device = device
        self.seed = seed

    def run(self):
        print("Scaffold job is runnning")
        rates=[self.rate]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []

        if self.cifar:
            image_dir = './images/cifar/SCAFFOLD/'
            figure_data_dir = './figure_data/cifar/SCAFFOLD/'
        elif self.mnist:
            image_dir = './images/mnist/SCAFFOLD/' 
            figure_data_dir = './figure_data/mnist/SCAFFOLD/'

        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(self.seed)
                if self.cifar: 
                    model = Cifar10CNN()
                elif self.mnist: 
                    model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

                self.agents.append(
                    Scaffold_Agent(
                        loss=nn.CrossEntropyLoss(),
                        model=model,
                        train_loader=loader,
                        epochs=self.epochs,
                        device=self.device,
                        lr=self.lr
                    )
                )
            
            torch.manual_seed(self.seed)
            if self.cifar:
                global_model = Cifar10CNN()
            elif self.mnist: 
                global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

            print("Setting the server...")
            server = DriftServer(
                clients=self.agents, 
                t_max=self.t_max, 
                model=global_model, 
                device=self.device,
                C=rate
            )
            print("All set, let's spin")
            server.spin(loader=self.val_loader)
        
            self.acc_per_rate[i,:] = server.val_accs
            self.rate_per_rate[i,:] = server.rates
            load = sum(self.rate_per_rate[i,:])/self.t_max
            acc, _ = server.validate_global(loader=self.test_loader)
            print(f'Test accuracy for rate {rate} = {acc}')
            self.loads.append(load)
            self.test_accs.append(acc.cpu().numpy())

            np.save(file=figure_data_dir+'accs_'+str(rate*100), arr=self.acc_per_rate)
            np.save(file=figure_data_dir+'loads_'+str(rate*100), arr=self.loads)

        print(f'saved: {self.acc_per_rate} and {self.loads}')



class FedLearnJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, prox: bool, device: str, epochs: int, rate: float,
                cifar: bool, mnist: bool, seed: int):
        self.cifar = cifar
        self.mnist = mnist
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.t_max = t_max
        self.lr = lr
        self.rate = rate
        self.rho = 0.01 if prox else 0
        self.device = device
        self.prox = prox
        self.seed = seed
        print(f'Prox: {self.prox}')

    def run(self):
        rates=[self.rate]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []

        if self.cifar:
            image_dir = './images/cifar/FedProx/' if self.prox else './images/cifar/FedAVG/'
            figure_data_dir = './figure_data/cifar/FedProx/' if self.prox else './figure_data/cifar/FedAVG/'
        elif self.mnist:
            image_dir = './images/mnist/FedProx/' if self.prox else './images/mnist/FedAVG/'
            figure_data_dir = './figure_data/mnist/FedProx/' if self.prox else './figure_data/mnist/FedAVG/'
        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(self.seed)
                if self.cifar: 
                    model = Cifar10CNN()
                elif self.mnist: 
                    model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
                self.agents.append(
                    FedLearn(
                        rho=self.rho,
                        loss=nn.CrossEntropyLoss(),
                        model=model,
                        train_loader=loader,
                        epochs=self.epochs,
                        device=self.device,
                        lr=self.lr
                    )
                )
            
            torch.manual_seed(self.seed)
            if self.cifar: 
                global_model = Cifar10CNN()
            elif self.mnist: 
                global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
            
            
            server = FedAgg(
                clients=self.agents, 
                t_max=self.t_max, 
                model=global_model, 
                device=self.device,
                C=rate
            )

            server.spin(loader=self.val_loader)
        
            self.acc_per_rate[i,:] = server.val_accs
            self.rate_per_rate[i,:] = server.rates
            load = sum(self.rate_per_rate[i,:])/self.t_max
            acc, _ = server.validate_global(loader=self.test_loader)
            print(f'Test accuracy for rate {rate} = {acc}')
            self.loads.append(load)
            self.test_accs.append(acc.cpu().numpy())

            np.save(file=figure_data_dir+'accs_'+str(rate*100), arr=self.acc_per_rate)
            np.save(file=figure_data_dir+'loads_'+str(rate*100), arr=self.loads)

        print(f'saved: {self.acc_per_rate} and {self.loads}')

class EventJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, num_agents: int, epochs: int,
                cifar: bool, mnist: bool, seed: int,
                comp_rate=None, forward=False, delta=None, rho=0.01) -> None:
        self.cifar = cifar
        self.mnist= mnist
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.forward = forward
        self.epochs = epochs
        self.rho = rho
        self.comp_rate = comp_rate
        self.rate=comp_rate
        self.delta = delta
        if delta is None or comp_rate is None: raise ValueError('Must give both delta and comp_rate')
        self.device = device
        self.num_agents = num_agents
        self.seed = seed

    def run(self):
        if self.forward: items = self.delta
        elif isinstance(self.rate, list): items = self.rate
        else: items = [self.rate]
        
        acc_per_item = np.zeros((len(items), self.t_max))
        rate_per_item = np.zeros((len(items), self.t_max))
        loads = []
        test_accs = []
        
        gamma = 0
        global_weight = self.rho/(self.rho*self.num_agents - 2*gamma)
        total_samples = sum([len(loader.dataset) for loader in self.train_loaders])

        if self.cifar: data_dir = 'figure_data/cifar/Event/'
        elif self.mnist: data_dir = 'figure_data/mnist/Event/'

        agents: List[TrainingAgentADMM] = []
        for j, loader in enumerate(self.train_loaders):
            data_ratio = len(loader.dataset)/total_samples            
            torch.manual_seed(self.seed)
            
            if self.cifar: 
                model = Cifar10CNN()
            elif self.mnist: 
                model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

            agents.append(
                TrainingAgentADMM(
                    N=len(self.train_loaders),
                    delta=self.delta,
                    rho=self.rho/(data_ratio*self.num_agents),
                    model=model,
                    loss=nn.CrossEntropyLoss(),
                    train_loader=loader,
                    epochs=self.epochs,
                    data_ratio=data_ratio,
                    device=self.device,
                    lr=self.lr,
                    global_weight=global_weight
                ) 
            )

        # Broadcast average to all agents and check if equal
        for agent in agents:
            agent.primal_avg = average_params([agent.model.parameters() for agent in agents])
        if self.device == 'cuda': torch.cuda.synchronize()

        torch.manual_seed(self.seed)
        if self.cifar:      
            global_model = Cifar10CNN()
        elif self.mnist:    
            global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)

        server = EventADMM(clients=agents, 
                           t_max=self.t_max, 
                           model=global_model, 
                           device=self.device, 
                           delta_z=self.delta/(100), 
                           comp_rate=self.comp_rate)
        server.spin(loader=self.test_loader)



 