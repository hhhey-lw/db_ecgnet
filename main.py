import argparse

from exp.exp_main import Exp_Main as Exp

parser = argparse.ArgumentParser(description='[Model] 12 Lead Electrocardiogram Classification')

parser.add_argument('--seed', type=int, default=2024, help='optimizer learning rate, [1, 10, 2024]')
parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU Training')
parser.add_argument('--gpu', type=int, default=0, help='GPU cuda:0')
parser.add_argument('--phase', type=str, default='train', help='train or test')

parser.add_argument('--experiment', type=str, default='rhythm', help='task of experiment, PTB-XL Options: [rhythm, superdiagnostic, all, diagnostic, subdiagnostic, form]')
parser.add_argument('--is_multi_label', type=bool, default=True, help='Task Type, multi-label or multi-class')
parser.add_argument('--model', type=str, default='DBECGNet', help='model of experiment, options: [DBECGNet], please check ./models/__init__.py')
parser.add_argument('--dataset', type=str, default='CPSC', help='dataset, options: [CPSC, chapman, ptbxl]')  # required=True
parser.add_argument('--datafolder', type=str, default='./data/dataset', help='root path of the data file. => ./data/CPSC')
parser.add_argument('--sampling_frequency', type=int, default='100', help='ecg signal sampling frequency')
parser.add_argument('--checkpoints_best', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--records_path', type=str, default='./result/', help='Location of saving model metircs results')

parser.add_argument('--train_epochs', type=int, default=80, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--milestones', type=list, default=[8, 32, 64], help='MultiStepLR')
parser.add_argument('--step_gamma', type=list, default=0.1, help='MultiStepLR')

args = parser.parse_args()
args.checkpoints_best = f'./checkpoints/{args.dataset}_{args.model}_{args.experiment}_checkpoint.pth'
args.records_path = f'./result/{args.dataset}_{args.model}_{args.experiment}_result.csv'
args.datafolder = f'./data/{args.dataset}/'
if args.dataset == 'chapman':
    args.is_multi_label = False

print('Args in experiment:')

exp = Exp(args)  # set experiments
print(f'Model:{args.model}, Dataset:{args.dataset}, Batch-Size:{args.batch_size}, Seed:{args.seed}')

# ===== The code was reorganized in the style of the "Informer" framework =====
# code: https://github.com/zhouhaoyi/Informer2020

if args.phase == 'train':
    print('>>>>>>> training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()
    print('>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
else:
    print('>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
