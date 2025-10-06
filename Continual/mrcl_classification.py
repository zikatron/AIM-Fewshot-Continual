import argparse
import logging
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import wandb
from utils.wandb_logger import init_wandb, finish_wandb

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

def setup_csv_logger(args):
    """Set up CSV file for logging training results."""
    # Create a more detailed filename that includes all fixed parameters
    csv_filename = f"{args.dataset}_{args.treatment}_frozen{args.rln}_tasks{args.tasks}_mlr{args.meta_lr}_ulr{args.update_lr}_ustep{args.update_step}_seed{args.seed}.csv"
    
    # Check if the file exists
    file_exists = os.path.isfile(csv_filename)
    
    # Open the CSV file in append mode
    csv_file = open(csv_filename, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write headers if the file is new
    if not file_exists:
        headers = [
            'step', 
            'unique_classes_seen', 
            'training_acc', 
            'test_acc'
        ]
        csv_writer.writerow(headers)
    
    return csv_file, csv_writer, csv_filename

def main(args):
    utils.set_seed(args.seed)

    if args.dataset == 'omniglot':
        args.classes = list(range(963))
    elif args.dataset == 'cifar100':
        args.classes = list(range(70))
    elif args.dataset == 'imagenet':
        args.classes = list(range(64))
        
    # Set up CSV logging
    csv_file, csv_writer, csv_filename = setup_csv_logger(args)
    print(f"Logging results to: {csv_filename}")
    
    # Initialize W&B
    config_wandb = vars(args)
    init_wandb(config=config_wandb, name=os.path.basename(csv_filename).split('.')[0])

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    config = mf.ModelFactory.get_model(args.treatment, args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config, args.treatment).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.saved_model, map_location='cpu')

        for idx in range(len(checkpoint)):
            maml.net.parameters()[idx].data = checkpoint.parameters()[idx].data

    maml = maml.to(device)

    utils.freeze_layers(args.rln, maml, args.treatment)
    cudnn.benchmark = True
    
    # Initialize a set to track unique classes seen
    unique_classes_seen = set()
    
    # Store test accuracy for logging
    test_accuracy = 0.0
    
    for step in range(args.steps):

        t1 = np.random.choice(args.classes, args.tasks, replace=False)
        
        # Update the set of unique classes seen
        unique_classes_seen.update(t1)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))
            maml.reset_classifer(t) #just commiting this out for now and to see if this is why it is matching

        d_rand_iterator = sampler.get_complete_iterator()
        accs, loss = maml(d_traj_iterators, d_rand_iterator)

        if step % 40 == 0:
            print('step: %d / %d   training acc %s' % (step, args.steps, str(accs)))
            
            # Log to CSV: training accuracy every 40 steps
            csv_writer.writerow([
                step,
                len(unique_classes_seen),
                accs,
                test_accuracy  # Use the last test accuracy or 0 if not available yet
            ])
            csv_file.flush()  # Make sure data is written to disk
            
            # Log to W&B
            wandb.log({
                "train_accuracy": accs,
                "test_accuracy": test_accuracy,
                "unique_classes_seen": len(unique_classes_seen)
            }, step=step)
            
        if step % 100 == 0 or step == args.steps - 1:
            torch.save(maml.net, '_'.join([args.dataset, args.treatment, str(step // 10000 * 10000) + '.net']))
            
        if step % 2000 == 0 and step != 0:
            # Get test accuracy
            test_accuracy = utils.log_accuracy(maml, iterator_test, device, step)
            train_accuracy = utils.log_accuracy(maml, iterator_train, device, step)
            
            # Log to CSV: test accuracy every 2000 steps
            csv_writer.writerow([
                step,
                len(unique_classes_seen),
                train_accuracy,
                test_accuracy
            ])
            csv_file.flush()  # Make sure data is written to disk
            
            # Log to W&B
            wandb.log({
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "unique_classes_seen": len(unique_classes_seen)
            }, step=step)
    
    # Close the CSV file when training is complete
    csv_file.close()
    
    # Finish W&B run
    finish_wandb()

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=20000)
    argparser.add_argument('--treatment', help='Model type', default='OML+AIM')
    argparser.add_argument('--checkpoint', help='Use a checkpoint model', action='store_true')
    argparser.add_argument('--saved_model', help='Saved model to load', default='my_model.net')
    argparser.add_argument('--seed', type=int, help='Seed for random', default=9)
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3) # 1e-2 from ANML
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2) # 0.01 from ANML
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=20)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=12)
    args = argparser.parse_args()

    print(args)
    main(args)
