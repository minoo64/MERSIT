import torch
###from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--task', help='GLUE Task: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli', 
					type=str, default='cola')
parser.add_argument('--net', help='LLMs: bert-base-cased, bert-large-cased, xlnet-base-cased, xlnet-large-cased', 
					type=str, default='bert-base-cased')
parser.add_argument('--gpus', help='list of used GPUs', type=str, default='0')

parser.add_argument('--qphase', help='train/calib/validation', type=int, default=2)
# 0: No
# 1/2/3/4/5     --> (1)INT8, (2)Normal-O-2A, (3)Multiple-O2A, (4)Truncation, (5)Clipping
# 6/7/8         --> (6)FL8, (7)POSIT8, (8)NewPOSIT8

parser.add_argument('--qnw', help='data bit-with', type=int, default=4)
parser.add_argument('--qna', help='data bit-with', type=int, default=4)

#Float/POSIT/MERSIT quant
parser.add_argument('--qm', help='mantissa', type=int, default=1)
parser.add_argument('--qe', help='exponent', type=int, default=3)
parser.add_argument('--qnosub', help='no subnormal in Floating-Point', action='store_true')

#O-2A quant
parser.add_argument('--o2aa', help='o2a activation bit-width', type=int, default=4)
parser.add_argument('--o2aw', help='o2a weight bit-width', type=int, default=4)
parser.add_argument('--o2ag', help='o2a group coding', type=int, default=4)

opt, _ = parser.parse_known_args()

class QInfo():
    def __init__(self, phase, data, qm=0, n=8, e=2, o2an=0, o2ag=0, nosub=False):
        self.phase = phase
        self.data = data
        self.n = n

        #FP/POSIT/MERSIT
        self.qm = qm
        self.e = e
        self.nosub = nosub
        #O-2A
        self.o2an = o2an
        self.o2ag = o2ag
