import torch
import argparse
import seaborn as sns
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from copy import copy
from admm.data import get_cifar_data, get_mnist_data
from cifar_jobs import FedLearnJob, FedADMMJob, SCAFFOLDJob, EventJob

#os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'  # Uncomment and set your WandB API key if needed
#import wandb

sns.set_theme()
num_gpus = 0
if torch.cuda.is_available(): 
    device = 'cuda'
    torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu = ''
    for i in range(num_gpus): gpu += f'{torch.cuda.get_device_name(i)}\n'
    print(gpu)
else:
    # device = 'cpu'
    raise Exception('GPU not available') 
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Example script with command line parsing.")
    parser.add_argument("--avg", action="store_true", default=False, help="Enable avg (default: False)")
    parser.add_argument("--cifar", action="store_true", default=False, help="Enable cifar (default: False)")
    parser.add_argument("--mnist", action="store_true", default=False, help="Enable mnist (default: False)")
    parser.add_argument("--prox", action="store_true", default=False, help="Enable prox (default: False)")
    parser.add_argument("--admm", action="store_true", default=False, help="Enable admm (default: False)")
    parser.add_argument("--back", action="store_true", default=False, help="Enable admm (default: False)")
    parser.add_argument("--scaffold", action="store_true", default=False, help="Enable scaffold (default: False)")
    parser.add_argument("--event", action="store_true", default=False, help="Enable event (default: False)")
    parser.add_argument("--seed", type=int, default=42, help="Set the seed")
    #if the method is event, then the delta threshold is required
    parser.add_argument("--delta", type=float, default=0, help="Set the delta threshold for Event")
    #otherwise rate is required
    if not parser.parse_known_args()[0].event:
        parser.add_argument("--rate", type=float, required=True, help="Set the rate")
        parser.add_argument("--comp_rate", type=float, default=0, help="Set the computation rate")
    else:
        parser.add_argument("--comp_rate", type=float, default=1, help="Set the computation rate for Event")
        parser.add_argument("--rate", type=float, default=0, help="Set the rate")

    args = parser.parse_args()
    print("avg:", args.avg)
    print("prox:", args.prox)
    print("admm:", args.admm)
    print("scaffold", args.scaffold)
    print("event", args.event)
    print("rate:", args.rate)
    print("delta:", args.delta)

    if args.avg:
        method='FedAvg'
        comp_rate = args.rate
        rate = args.rate
    elif args.prox:
        method='FedProx'
        comp_rate = args.rate
        rate = args.rate
    elif args.admm:
        method='FedADMM'
        comp_rate = args.rate
        rate = args.rate
    elif args.scaffold:
        method='SCAFFOLD'
        comp_rate = args.rate
        rate = args.rate
    elif args.event:
        method = 'Event'
        comp_rate = args.comp_rate
        rate = comp_rate
    else:
        raise ValueError('Need to select an algorithm (avg, prox, admm, scaffold)')
    
    if args.cifar:
        dataset_name='CIFAR-10'
    elif args.mnist:
        dataset_name='MNIST'
    else:
        raise ValueError('Need to select a dataset (cifar, mnist)')
    delta = args.delta
    '''
    wandb.init(
        # set the wandb project where this run will be logged
        # project="project_name",
        # track hyperparameters and run metadata
        config={
        "algorithm": method,
        "dataset": dataset_name,
        "rate": rate,
        "delta": delta,	
        "seed": args.seed,
        }
    )
    print("WANDB is set.")
    '''
    # Check for exclusive flags
    if sum([args.avg, args.prox, args.admm, args.back, args.scaffold, args.event]) != 1:
        parser.error("Exactly one of --avg, --prox, --event, or --admm must be set to True.")
    if args.cifar and args.mnist: parser.error('Can only specify one experiment, eith cifar or mnist')
    if not args.cifar and not args.mnist: parser.error('Must specify eith cifar or mnist')
    print("No flag 1")
    if args.cifar:
        num_clients = 100
        batch_size = 20
        train_loaders, test_loader = get_cifar_data(num_clients=num_clients, batch_size=batch_size, seed=args.seed)
        val_loader = copy(test_loader)
        rho = 0.01
        lr=0.01
    elif args.mnist:
        num_clients = 10
        # choose a large batch size to reduce the number of iterations
        batch_size = 256
        lr=0.01
        train_loaders, test_loader, val_loader = get_mnist_data(num_clients=num_clients, batch_size=batch_size, seed=args.seed)
        val_loader = copy(test_loader)
        rho = 1
    print("dataloaders are ready")
    """
    Run Experiments
    """
   
    if args.cifar:
        t_max = 500
        epochs = 3
    if args.mnist:
        t_max = 150
        epochs = 1
    print("Parse Arguments to Jobs")
    
    prox_args = {
        'train_loaders': train_loaders, 'test_loader': test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr': lr, 'device': device, 'prox': True, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist, 'seed': args.seed
    }
    avg_args = prox_args.copy()
    avg_args['prox'] = False

    ADMM_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr':lr, 'device':device, 'num_agents':num_clients, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist,  'seed': args.seed
    }

    scaffold_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr':lr, 'device':device, 'num_agents':num_clients, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist,  'seed': args.seed
    }

    event_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr':lr, 'device':device, 'num_agents':num_clients, 'epochs': epochs, 'delta': delta,
        'cifar': args.cifar, 'mnist':args.mnist,  'seed': args.seed, 'comp_rate': comp_rate, 'rho': rho
    }

    print("checkpoint 1")
    if args.prox: job = FedLearnJob(**prox_args)
    elif args.avg: job = FedLearnJob(**avg_args)
    elif args.admm: job = FedADMMJob(**ADMM_args)
    elif args.scaffold: job = SCAFFOLDJob(**scaffold_args)
    elif args.event: job = EventJob(**event_args)
    else: raise ValueError('Need to select an algorithm (avg, prox, admm, SCAFFOLD, Event)')
    print("checkpoint 2")
    job.run()