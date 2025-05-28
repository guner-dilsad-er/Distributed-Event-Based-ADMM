import torch
from typing import List
from admm.utils import sum_params, add_params, average_params, difference_params
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from collections import OrderedDict
import statistics
import subprocess
import re
from admm.utils import sublist_by_fraction
import copy

#import wandb


class ServerBase:

    def __init__(self, t_max: int, model: torch.nn.Module, device: str) -> None:
        self.t_max = t_max
        self.pbar = tqdm(range(t_max))
        self.device = device
        self.global_model = model.to(self.device)
        self.global_params = self.get_parameters(self.global_model)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def set_parameters(self, parameters, model: torch.nn.Module) -> None:
        """Change the parameters of the model using the given ones."""
        model_copy = copy.deepcopy(model)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).clone().detach() for k, v in params_dict})
        model_dict = model.state_dict()
        new_state_dict={k:v if v.size()==model_dict[k].size()  else  model_dict[k] for k,v in zip(model_dict.keys(), state_dict.values())}
     
        model_copy.load_state_dict(new_state_dict, strict=False)
        return model_copy
    
    def get_parameters(self, model):
        """Return the parameters of the current net."""
        model = copy.deepcopy(model)
        copied_params = [param.clone().detach().to(self.device) for param in model.parameters()]
        return copied_params

    def validate_global(self, loader: DataLoader) -> float:
        wrong_count = 0
        total = len(loader.dataset)
        validation_loss = 0
        self.global_model.eval()
        for data, target in loader:
            
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            output= self.global_model(data)
            out = torch.argmax(output, dim=1)
            wrong_count += torch.count_nonzero(out-target)
            # Classification loss on the validation set
            validation_loss += self.loss_function(output, target).item()
        # Classification accuracy on the validation set
        global_acc = 1 - wrong_count/total
        # Average loss on the validation set
        validation_loss /= len(loader)
        return global_acc, validation_loss

class EventADMM(ServerBase):

    def __init__(self, clients: List[agents.TrainingAgentADMM], t_max: int, model: torch.nn.Module, device: str, delta_z: float, comp_rate:float) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        
        self.comm = 0
        self.comm_z= 0
        self.N = len(self.agents)
        self.C=comp_rate
        self.p_rate = comp_rate
        self.delta_z = delta_z
        # For experiment purposes
        self.agent_comm_load = []
        self.overall_comm_load = []
        self.val_accs = []
        self.val_loss = []
        self.local_res = [self.get_parameters(self.global_model) for _ in range(self.N)]

        self.primal_avg = [torch.zeros_like(param) for param in self.global_model.parameters()]
        self.accumulated_residuals = [torch.zeros_like(param) for param in self.global_model.parameters()]

    def spin(self, loader=None) -> None:
        # Sample subset of agents as lucky agents
        # lucky_agents = sublist_by_fraction(agents=self.agents, fraction=self.p_rate)
        self.sent_global_params = self.get_parameters(self.global_model)
        self.sent_global_model = self.global_model
        sampled_agents = self.agents

        for agent in self.agents:
            agent.receive = False

        for round in self.pbar:
            
            # Primal Update
            D = []
            for agent in self.agents:
                agent.broadcast = False

            for agent in sampled_agents:
                if agent.receive:
                    current_global_params= self.get_parameters(self.global_model)
                    d = agent.primal_update(round, params=current_global_params)
                    D.append(d)
                    agent.receive = False
                else:
                    d = agent.primal_update(round, params=self.sent_global_params)
                    D.append(d)

            delta_description = f', min Change: {min(D):.5f}, max Change: {max(D):.5f}, Median Change: {statistics.median(D):.5f}'
            if self.device == 'cuda': torch.cuda.synchronize()
            
            # Residual update in the case of communication
            comm_list=[]
            self.comm = 0
            #for i, agent in enumerate(self.agents):
            for i, agent in enumerate(sampled_agents):
                if agent.broadcast: 
                    comm_list.append(i)
                    self.comm += 1
                    add_params(self.local_res[i], agent.residual)
                    agent.broadcast=False
            #self.global_params = average_params(self.local_res)
            self.primal_avg = average_params(self.local_res)
            # Now check how is the primal_avg (server params) different than the global model (broadcasted)
            change_server = difference_params(self.primal_avg, self.sent_global_model.parameters())
            # change_server is the difference between the global model and the primal_avg
            # change_server : list of tensors
            # Print to see the change
            #print(f'Change: {torch.norm(change_server)}')
            # Compare its norm with delta_z
            self.global_model = self.set_parameters(self.primal_avg, self.global_model)
            self.global_params = self.get_parameters(self.global_model)
            
            if torch.norm(change_server) >= self.delta_z:     
                self.comm_z += 1*len(sampled_agents)
                # If the change is larger than delta_z, then broadcast residuals to agents
                self.sent_global_params = self.get_parameters(self.global_model)
                self.sent_global_model = self.global_model
                #for agent in agents:
                #    agent.receive = True
            else:
                self.comm_z += 0
                # If the change is smaller than delta_z, then do not broadcast the global model to all agents
                # pick a lucky portion of agents and broadcast the global model to them
                lucky_agents = sublist_by_fraction(agents=self.agents, fraction=self.p_rate)
                for agent in lucky_agents:
                    agent.receive = True
                    self.comm_z += 1
                # print the size of lucky_agents
                print(f'Lucky agents size: {len(lucky_agents)}')
                

            # Test updated params on validation set
            acc_descrption = ''
            if loader is not None:
                # Get gloabl variable Z and copy to a network for validation
                with torch.no_grad():
                    global_acc, global_loss = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}'
            #wandb.log({"val_acc": global_acc, "val_loss": global_loss, "delta_z": self.delta_z, "run_load": self.comm, "run_load_z": self.comm_z})

            print(f'Round {round+1}, Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}, Comm Cost = {self.comm+self.comm_z}')

            if self.device == 'cuda': torch.cuda.synchronize()
            
            if self.device == 'cuda':
                command = 'nvidia-smi'
                p = subprocess.check_output(command)
                ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
                GPU_desctiption = f', ram = {ram_using}'
            else: GPU_desctiption = ''
            
            # Analyse communication frequency
            freq = self.comm/(self.N)
            agent_comm=self.comm
            z_comm =self.comm_z
            self.comm = 0
            self.comm_z = 0
            self.pbar.set_description(f'Agent Comm: {freq:.3f}, Server Comm: {z_comm:.1f} ' + acc_descrption + delta_description + GPU_desctiption)

            # For experiment purposes
            self.overall_comm_load.append(agent_comm + z_comm)
            self.agent_comm_load.append(agent_comm)
            self.val_accs.append(global_acc.detach().cpu().numpy())
            self.val_loss.append(global_loss)


        self.overall_comm_load = np.array(self.overall_comm_load)
        self.agent_comm_load = np.array(self.agent_comm_load)
        self.val_accs = np.array(self.val_accs)
        self.val_loss = np.array(self.val_loss)

    def validate_agents(self, loader: DataLoader) -> List[float]:
        total = 0
        wrong_count = np.zeros(self.N)
        for data, target in loader:
            total += target.shape[0] 
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    data, target = data.to(agent.device), target.type(torch.LongTensor).to(agent.device)
                    output = agent.model(data)
                    out = torch.argmax(output, dim=1)
                    wrong_count[i] += torch.count_nonzero(out-target)
        model_accs = [1 - wrong/total for wrong in wrong_count]
        # cross entropy loss
        model_losses = [agent.loss(agent.model(data.to(agent.device)), target.type(torch.LongTensor).to(agent.device)).item() for agent in self.agents]
        return model_accs, model_losses
    


    

class InexactADMM(ServerBase):
    
    def __init__(self, clients: List[agents.FedADMM], C: float, t_max: int, 
                model: torch.nn.Module, device: str, num_clients: int, k0: int) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.C = C
        self.N = len(self.agents)
        self.k0 = k0
        # For experiment purposes
        self.rates = []
        self.val_accs = []
        self.num_clients = num_clients


    def spin(self, loader=None) -> None:
        sampled_agents = self.agents
        for round in self.pbar:
            
            # Collect params from sublist of clients 
            if round%self.k0==0:
                # Sample subset of agents
                sampled_agents = sublist_by_fraction(agents=self.agents, fraction=self.C)
                self.comm += len(sampled_agents)
                global_params = average_params([agent.residual for agent in self.agents])

            for clients in sampled_agents:
                clients.update(global_params=global_params)
            if self.device == 'cuda': torch.cuda.synchronize()

            self.global_model = self.set_parameters(global_params, self.global_model)
            
            # Validate global model
            acc_descrption = ''
            if loader is not None:
                with torch.no_grad():
                    global_acc, global_loss = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'
            #wandb.log({"val_acc": global_acc, "val_loss": global_loss})
            print(f'Round {round+1}, Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}, Comm Cost = {self.comm*2}')

            # Analyse communication frequency
            freq = self.comm/(self.N)
            self.comm = 0
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)


class FedAgg(ServerBase):
    def __init__(self, clients: List[agents.FedLearn], C: float, t_max: int,
                  model: torch.nn.Module, device: str) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.C = C
        self.N = len(self.agents)
        # For experiment purposes
        self.rates = []
        self.val_accs = []
        

    def spin(self, loader=None) -> None:

        for round in self.pbar:

            # Sample subset of agents
            sampled_agents = sublist_by_fraction(agents=self.agents, fraction=self.C)
            global_params = self.get_parameters(self.global_model)
           
            self.comm = len(sampled_agents)
            with torch.no_grad():
                global_acc, global_loss = self.validate_global(loader=loader)
                print(f'Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}')

            # Send params to clients and let them train
            m_t = 0
            weighted_local_params = []
            for agent in sampled_agents:
                agent.primal_update(global_params)
                m_t += agent.num_samples
            if self.device == 'cuda': torch.cuda.synchronize()
            
            # Aggregate the new params
            for agent in sampled_agents:
                weighted_local_params.append([param*agent.num_samples/m_t for param in agent.model.parameters()])
            global_params = sum_params(weighted_local_params)
            self.global_model = self.set_parameters(global_params, self.global_model)

            # Validate global model
            acc_descrption = ''
            if loader is not None:
                with torch.no_grad():
                    global_acc, global_loss = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'
            #wandb.log({"val_acc": global_acc, "val_loss": global_loss})
            print(f'Round {round+1}, Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}, Comm Cost = {self.comm*2}')

             # Analyse communication frequency
            freq = self.comm/(self.N)
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)


        
class DriftServer(ServerBase):
    "Server for the Scaffold Method" 
    def __init__(self, clients: List[agents.FedLearn], C: float, t_max: int,
                  model: torch.nn.Module, device: str) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.C = C
        self.N = len(self.agents)
        self.global_c = [torch.zeros(param.shape).to(self.device) for param in self.agents[0].model.parameters()]
        # For experiment purposes
        self.rates = []
        self.val_accs = []

    def spin(self, loader=None) -> None:
        
        for round in self.pbar:
            
            # Sample subset of agents
            sampled_agents = sublist_by_fraction(agents=self.agents, fraction=self.C)
            global_params = self.get_parameters(self.global_model)
            global_c = self.global_c
            self.comm = len(sampled_agents)
            #print("Before Primal Update")

            # Send params to clients and let them train
            m_t = 0
            # check if sampled_agents is empty
            if len(sampled_agents) == 0:
                raise ValueError("Sampled agents is empty")

            for agent in sampled_agents:
                agent.primal_update(global_params, global_c)
                m_t += agent.num_samples
            if self.device == 'cuda': torch.cuda.synchronize()
            # Agents calculate delta_y_i and delta_c_i
            #print(m_t)
            #print("Primal Update Done")
            # collect delta_y_i and delta_c_i and sum them
            # Aggregate the new params
            collection_delta_y_i = []
            collection_delta_c_i = []
            n_g = 1

            for agent in sampled_agents:
                collection_delta_y_i.append([param*agent.num_samples/m_t*1/n_g for param in agent.delta_y_i])
                collection_delta_c_i.append([param*agent.num_samples/m_t*1/self.N for param in agent.delta_c_i])
            delta_x = sum_params(collection_delta_y_i)
            delta_c = sum_params(collection_delta_c_i)  
        
            # c = c +1/N delta_c
    
            # add_params does not work, add delta_c to global_c explicitly
            for i in range(len(global_c)):
                global_c[i] = global_c[i] + delta_c[i]
            
            
            self.global_c = global_c

            # x = x + 1/n_g delta_x
            # add_params does not work, add delta_x to global_params explicitly
            for i in range(len(global_params)):
                global_params[i] = global_params[i] + delta_x[i]

            self.global_model = self.set_parameters(global_params, self.global_model)
            # Validate global model
            acc_descrption = ''
            if loader is not None:
                with torch.no_grad():
                    global_acc, global_loss = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'
            #wandb.log({"val_acc": global_acc, "val_loss": global_loss})
            print(f'Round {round+1}, Global Acc = {global_acc:.4f}, Global Loss = {global_loss:.4f}, Comm Cost = {self.comm*2}')

            # Analyse communication frequency
            freq = self.comm/(self.N)
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)