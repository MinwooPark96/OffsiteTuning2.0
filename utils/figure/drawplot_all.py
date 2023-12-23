from drawplot_epoch_acc import drawplot_epoch_acc
from drawplot_train_loss import drawplot_train_loss
from drawplot_valid_acc import drawplot_valid_acc
from drawplot_valid_loss import drawplot_valid_loss
from drawplot_epoch_loss import drawplot_epoch_loss

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--length','-l',type = int, default = None)
    parser.add_argument('--json_name','-n',type = str, default = None)
    parser.add_argument('--nbins','-b',type = int, default = 10)
    
    
    args = parser.parse_args()
    
    length = args.length
    json_name = args.json_name
    nbins = args.nbins
    
    if not json_name:
        print("insert json_name in result directory!")
    
    drawplot_valid_loss(length,json_name,nbins)    
    drawplot_valid_acc(length,json_name,nbins)    
    drawplot_train_loss(length,json_name,nbins)    
    drawplot_epoch_acc(length,json_name,nbins)    
    drawplot_epoch_loss(length,json_name,nbins)    
        
