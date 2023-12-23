import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from src.config_parser import create_config
from tools.valid_tool import valid

from utils import utils

# format='%(asctime)s - %(levelname)s - %(name)s - %(message)s - %(lineno)d'

logging.basicConfig(format='%(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    # filename="./log/train_cross.log",
                    # filemode='w')
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    
    parser.add_argument("--prompt_emb", type=str, default=None)
    parser.add_argument("--projector", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, default="valid")
    
    parser.add_argument("--output_name", type=str, default=None)
    
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)
    
    os.system("clear")

    if args.prompt_emb :
        config.set("train","prompt_emb",args.prompt_emb)
        print("prompt_emb",config.get("train","prompt_emb"))
    
    else :
        logger.warning("please set prompt_emb! e.g. SST2PromptBert")
        exit()
    
    
    gpu_list = utils.set_gpu(config,args)
        
    local_rank = config.getint('distributed', 'local_rank')
    
    
    if local_rank <= 0:
        logger.info("config file = <{}>".format(configFilePath))
        
    
    cuda = torch.cuda.is_available()
    cuda_available = str(cuda)
    
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    
    utils.set_random_seed(args.seed)

    assert len(config.get("data","valid_dataset_type").lower().split(',')) == 1
    
    parameters = init_all(config, gpu_list, args.mode, local_rank = local_rank, args=args)
    
    model = parameters["model"]

    valid(model, parameters["valid_dataset"], 1, config, gpu_list, parameters["output_function"], mode=args.mode, args=args)
    