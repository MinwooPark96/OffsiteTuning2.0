from torch.utils.data import DataLoader
import logging

import src.formatter as form
from src.dataset import dataset_list
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

collate_fn = {} #collate_fn["train"] = formatter["train"].process = {"inputx", "mask", "label", "mode_name"}
formatter = {}

def init_formatter(config, mode, *args, **params):
    
    local_rank = config.getint("distributed","local_rank")
    
        
    formatter[mode] = form.init_formatter(config, mode, *args, **params)
    
    def train_collate_fn(data):
        return formatter["train"].process(data, config, "train", args=params)

    def valid_collate_fn(data):
        return formatter["valid"].process(data, config, "valid")#, args=params)

    def test_collate_fn(data):
        return formatter["test"].process(data, config, "test")#, args=params)

    if mode == "train":
        collate_fn[mode] = train_collate_fn
    elif mode == "valid":
        collate_fn[mode] = valid_collate_fn
    else:
        collate_fn[mode] = test_collate_fn
    
    

def init_one_dataset(config, mode, data_name, *args, **params):
    temp_mode = mode
    local_rank = config.getint("distributed","local_rank")
    local_map = config.getboolean("train","local_map")
    if local_map:
        which_data = 'crossLocalmap'
    else:
        which_data = "cross" 
    
    dataset = dataset_list[which_data](config, mode, data_name, *args, **params) #dataset.py
    
    batch_size = config.getint("train", "batch_size")
    
    reader_num = config.getint("train", "reader_num") # data loading process number
    drop_last = True #train,valid 는 잔여 batch 버림.
    
    # evaluation mode 일 경우
    if mode in ["valid", "test"]:
        
        if mode == "test": #test 시에는 잔여 batch 를 버리지 않음
            drop_last = False

        try:
            batch_size = config.getint("eval", "batch_size")
            
        except Exception as e:
            logger.warning("[eval] batch size has not been defined in config file, use [train] batch_size instead.")
                
        try:
            reader_num = config.getint("eval", "reader_num")
            
        except Exception as e:
            logger.warning("[eval] reader_num has not been defined in config file, use [train] reader num instead.")
            
            
    if config.getboolean('distributed', 'use'):
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            #shuffle=shuffle,
                            num_workers=reader_num, #데이터 로딩하는 process number.
                            collate_fn=collate_fn[mode], #collate_fn = formatter.process
                            drop_last=drop_last,
                            sampler=sampler)
    
    return dataloader
    
# test 용
def init_test_dataset(config, *args, **params):
    init_formatter(config, ["test"], *args, **params)
    local_rank = config.getint("distributed","local_rank")

    test_dataset = init_one_dataset(config, "test", *args, **params)
    #logger.info("[minwoo] test_dataset = {}".format(test_dataset.__dict__.keys()))
    return test_dataset

#parameters["train_dataset"], parameters["valid_dataset"] = init_dataset(config, *args, args=kwargs) #train_tool.py
def init_dataset(config, *args, **params):
    local_rank = config.getint("distributed","local_rank")
    
    
    init_formatter(config, "valid", *args, **params) 
    
    valid_dataset_list = config.get("data","valid_dataset_type").lower().split(',')
    
    if len(valid_dataset_list) >= 2 :    
        valid_dataset = [init_one_dataset(config,"valid",one_data_name,*args, **params) for one_data_name in valid_dataset_list]
    else :
        if 'only_valid' in params and params['only_valid']:
            valid_dataset = init_one_dataset(config,"valid",config.get("data","valid_dataset_type"),*args, **params)

        else :
            valid_dataset = [init_one_dataset(config,"valid",config.get("data","valid_dataset_type"),*args, **params)]
        
    if local_rank <= 0:
        logger.info("valid data loader is setted!")
    
    if 'only_valid' in params and params['only_valid']:
        return valid_dataset

    init_formatter(config, "train", *args, **params) 
    
    train_dataset_list = config.get("data","train_dataset_type").lower().split(',')
    
    # if len(train_dataset_list) >= 2 :
    #     train_dataset = [init_one_dataset(config,"train",one_data_name,*args, **params) for one_data_name in train_dataset_list]
    # else :
    #     train_dataset = init_one_dataset(config,"train",config.get("data","train_dataset_type"),*args, **params)
    
    train_dataset = [init_one_dataset(config,"train",one_data_name,*args, **params) for one_data_name in train_dataset_list]
    
    
    if local_rank <= 0:
        logger.info("train data loader is setted!")
        
    return train_dataset, valid_dataset

if __name__ == "__main__":
    pass


    