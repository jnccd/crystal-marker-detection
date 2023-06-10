import argparse
from pathlib import Path

from cmd_tf.training import fit

def main():
    parser = argparse.ArgumentParser(prog='ml-segmentation', description='Trains a sample network on the synthetic generated data.')
    parser.add_argument('-bs','--batch-size',       type=int,               default=32,         help='The batch size of the training data.')
    parser.add_argument('-e','--epochs',            type=int,               default=1,          help='The number of epochs to learn for.')
    parser.add_argument('-pm','--print-model',      action='store_true',    default=False,      help='Print the model summary of this run.')
    parser.add_argument('-r','--run',               type=str,               default='default',  help='The name of the run to learn in.')
    parser.add_argument('-mgs','--multi-gpu-strategy', action='store_true', default=False,      help='Use the tensorflow strategy for multi gpu learning.')
    parser.add_argument('-df','--train-data-folder',type=str,               default='renders',  help='The trainings data folder name to learn from or build into.')
    # TODO: Maybe readd this later
    #parser.add_argument('-av','--analyze-valdata',  action='store_true',    default=False,      help='Instead of learning, compute metrics for already written validation data.')
    args = parser.parse_args()
    
    
    fit(
        batch_size=args.batch_size, 
        num_epochs=args.epochs, 
        run=args.run, 
        data_folder=args.train_data_folder, 
        size=args.size,
        print_model=args.print_model,
        use_multi_gpu_strategy=args.multi_gpu_strategy,
        )