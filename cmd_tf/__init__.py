import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='cmd-tf', description='Trains a sample network on the synthetic generated data.')
    
    parser.add_argument('-bs','--batch-size',       type=int,               default=16,         help='The batch size of the training data.')
    parser.add_argument('-e','--epochs',            type=int,               default=1,          help='The number of epochs to learn for.')
    parser.add_argument('-s','--size',              type=int,               default=320,        help='The image size the network should read.')
    parser.add_argument('-pm','--print-model',      action='store_true',    default=False,      help='Print the model summary of this run.')
    parser.add_argument('-r','--run',               type=str,               default='default',  help='The name of the run to learn in.')
    parser.add_argument('-mgs','--multi-gpu-strategy', action='store_true', default=False,      help='Use the tensorflow strategy for multi gpu learning.')
    parser.add_argument('-df','--dataset-folder',   type=str,               default='',         help='The trainings data folder name to learn from or build into.')
    parser.add_argument('-da','--data-augmentation',action='store_true',    default=False,      help='If set will enable data augmentation in the dataloaders if implemented.')
    
    parser.add_argument('-t','--test',              action='store_true',    default=False,      help='Instead of learning, test run config model on other data.')
    parser.add_argument('-td','--testdata',         type=str,               default='',         help='The test data folder name or file name to get the testdata from.')
    
    args = parser.parse_args()
    
    extra_settings = {
        'data_aug': args.data_augmentation
    }
    
    if args.test:
        from cmd_tf.test import test
        test(
            run=args.run,
            size=args.size,
            testdata=args.testdata,
            extra_settings=extra_settings,
            )
    else:
        from cmd_tf.training import fit
        fit(
            batch_size=args.batch_size, 
            num_epochs=args.epochs, 
            run=args.run, 
            data_folder=args.dataset_folder, 
            size=args.size,
            print_model=args.print_model,
            use_multi_gpu_strategy=args.multi_gpu_strategy,
            extra_settings=extra_settings,
            )