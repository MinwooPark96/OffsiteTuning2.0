import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from src.config_parser import create_config

from tools.train_tool_cross_mtl import train as train
from utils import utils
from utils import downloadingPLM

logging.basicConfig(format='%(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    # filename="./log/train_cross.log",
                    # filemode='w')
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 필수 argument
    
    parser.add_argument('--config', '-c', help="specific config file", default='src/config/default.config') 
    parser.add_argument('--gpu', '-g', help="gpu id list", default='0') # -> "0"
    parser.add_argument("--prompt_emb",type=str,default=False) 
    parser.add_argument("--source_model",type=str,default=False) 
    
    parser.add_argument('--do_test', help="do test while training or not", action="store_true",default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_transfer", type=str, default=False)
    
    os.system("clear")
    
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    if args.prompt_emb and args.source_model :
        logger.warning("set only one argument <prompt_emb> or <source_model>")        
        exit()
    elif (not args.prompt_emb) and (not args.source_model):
        logger.warning("set at least one argument <prompt_emb> or <source_model>")        
        exit()
    
    config.set("train","prompt_emb",args.prompt_emb)
    config.set("train","source_model",args.source_model)

    print("prompt_emb",config.get("train","prompt_emb"))
    print("source_model",config.get("train","source_model"))
    
    
    gpu_list = utils.set_gpu(config,args)
    
    local_rank = config.getint('distributed', 'local_rank')
    
    torch.distributed.barrier()
    
    if local_rank <= 0:
        logger.info("config file = <{}>".format(configFilePath))
    
    cuda = torch.cuda.is_available()
    cuda_available = str(cuda)
    
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    
    utils.set_random_seed(args.seed)

    parameters = init_all(config, gpu_list, "train", local_rank = local_rank, args=args) #model_prompt=args.source_model 제거
    
    do_test = False
    
    if args.do_test:
        do_test = True
    
    torch.distributed.barrier()
    
    train(parameters, config, gpu_list, do_test,local_rank, args=args)
    
    
    
    
