import logging
import os
import torch
import logging
import random
import numpy as np
from typing import Optional

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer, AE_1_layer_mutiple_100_paper,AE_transformer_layer,AE_1_layer_tokenwise,AE_auto_layer    


import torch.optim as optim
from transformers import AdamW

import torch
    
def set_gpu(config,args)->list:
    """set gpu, local_rank, and distributed.init_process"""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_list = list(map(int,args.gpu.strip().split(",")))
    gpu_list = list(range(0,len(device_list)))
    
    if len(gpu_list) >= 2:
        local_rank = int(os.environ["LOCAL_RANK"])
        config.set('distributed', 'local_rank', local_rank)
    
    else:
        config.set('distributed', 'local_rank', -1)
    
    if config.getboolean("distributed", "use") and len(gpu_list)>1:
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        torch.cuda.set_device(gpu_list[local_rank])
        config.set('distributed', 'gpu_num', len(gpu_list))
    
    else:
        config.set("distributed", "use", False)
    
    
    return gpu_list
    
    
    
def set_random_seed(seed:Optional[int]) -> None:
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
def save_projector(filename,model_AE):
    filename = filename.strip().replace(".pkl","")
    filename = filename+"_model_cross.pkl"
    try:
        torch.save(model_AE.state_dict(), filename)
        print("save projector ... <{}> .".format(filename))
    
    except Exception as e:
        print("Fail to save projector")

def load_projector(config):
    
    projector = config.get("projector","projector")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'AE_1' in projector:
        if config.getboolean("projector","flatten") : 
            dim_0,dim_1,dim_2 = config.getint("projector","dim_0"),config.getint("projector","dim_1"),config.getint("projector","dim_2")
            model_AE = AE_1_layer_mutiple_100(dim_0=dim_0,dim_1=dim_1,dim_2=dim_2).to(device)
        else :
            if 'auto' in projector:
                    values = list(map(int,config.get('projector','dims').strip().split(',')))
                    keys = ["dim_"+str(idx) for idx in range(len(values))]
                    dims = dict(zip(keys,values))
                    model_AE = AE_auto_layer(**dims).to("cuda")

            elif 'transformer' in projector:
                    dim_0,dim_1= config.getint("projector","dim_0"),config.getint("projector","dim_1")
                    model_AE = AE_transformer_layer(dim_0=dim_0,dim_1=dim_1).to(device)
            else:
                dim_0,dim_1,dim_2 = config.getint("projector","dim_0"),config.getint("projector","dim_1"),config.getint("projector","dim_2")
                model_AE = AE_1_layer(dim_0 = dim_0, dim_1 = dim_1, dim_2 = dim_2).to(device)
    else:
        print("Fail to select projector.")
        NotImplementedError
    
    return model_AE

def init_projector(config):
    
    projector = config.get("projector","projector")
    
    #phase1 : projector model
    model_AE = load_projector(config)
    
    #phase2 : init
    if config.getint('distributed','local_rank') <=0 :
        print("selected projector class is <{}>".format(projector))
        
    for module in model_AE.modules():
        if isinstance(module, torch.nn.Linear):
            if "roberta" in config.get("target_model","model_base").lower():
                pass
            elif "t5" in config.get("target_model","model_base").lower():
                torch.nn.init.normal_(module.weight, mean=0, std=1)
            else:
                torch.nn.init.normal_(module.weight, mean=0, std=1)

    #phase3 : load checkpoint
    checkpoint_dir= "checkpoint/"+config.get("output", "model_name") 
    
    if os.path.isdir(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            last_checkpoint = checkpoints[0] 
            for checkpoint_name in checkpoints:
                checkpoint_epoch = int(checkpoint_name.split("_")[0])
                last_checkpoint_epoch = int(last_checkpoint.split("_")[0])
                if checkpoint_epoch >= last_checkpoint_epoch:
                    last_checkpoint = checkpoint_name
                    
            model_AE.load_state_dict(torch.load(checkpoint_dir+"/"+last_checkpoint, map_location=lambda storage, loc:storage))
            
            if config.getint('distributed','local_rank') <=0 :
                print("Load trained weight of projector from <{}>".format(checkpoint_dir+"/"+last_checkpoint))
                
    return model_AE

def loadPLM(config,target_model = True):
    local_map = config.getboolean("train","local_map")
    assert local_map == False
    
    if target_model:    
        model_name = config.get("target_model","model_base").lower() 
    else :
        model_name = config.get("train","source_model").lower()
    
    if "roberta" in model_name:
        try:
            if "large" in model_name:
                model = "roberta-large"
                ckp = "PLM/RobertaLargeForMaskedLM"
                hidden_size = 1024
            else:
                model = "roberta-base"
                ckp = "PLM/RobertaForMaskedLM"
                hidden_size = 768
            
        except:
            model = "roberta-base"
            ckp = "PLM/RobertaForMaskedLM"
            hidden_size = 768
    
    elif "bert" in model_name:
        try:
            if "large" in model_name:
                model = "bert-large"
                ckp = "PLM/BertLargeForMaskedLM"
                hidden_size = 1024
            elif "medium" in model_name:
                model = "prajjwal1/bert-medium"
                ckp = "PLM/BertMediumForMaskedLM"
                hidden_size = 512
            else :
                model = "bert-base-uncased"
                ckp = "PLM/BertForMaskedLM"
                hidden_size = 768
            
        except:
            model = "bert-base-uncased"
            ckp = "PLM/BertForMaskedLM"
            hidden_size = 768
    elif 't5' in model_name :
        
        try:
            if "small" in model_name:
                model = "t5-small"
                ckp = "PLM/T5SmallForMaskedLM"
                hidden_size = 512
            elif "large" in model_name:
                model = "t5-large"
                ckp = "PLM/T5LargeForMaskedLM"
                hidden_size = 1024
            elif "b3" in model_name:
                model = "t5-b3"
                ckp = "PLM/T5B3ForMaskedLM"
                hidden_size = 1024
            
            else:
                model = "t5-base"
                ckp = "PLM/T5ForMaskedLM"
                hidden_size = 768
        except:
            model = "t5-base"
            ckp = "PLM/T5ForMaskedLM"
            hidden_size = 768

    
    else:
        
        print("load_model function error! <{}>".format(model_name))
        exit()
    
    if "bert-medium" in model: 
        model = "bert-medium"

    plmconfig = AutoConfig.from_pretrained(model)
    plmconfig.prompt_num = config.getint("prompt", "prompt_num")
    plmconfig.prompt_len = config.getint("prompt", "prompt_len")

    
    if "large" in model_name:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Large"+"_init_params"
    
    elif "medium" in model_name:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Medium"+"_init_params"
    
    else:
        init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"_init_params"

    if "roberta" in model_name:
        from src.model.modelling_roberta import RobertaForMaskedLM
        encoder = RobertaForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
    elif "bert" in model_name:
        from src.model.modelling_bert import BertForMaskedLM
        encoder = BertForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
    
    elif 't5' in model_name:
        from src.model.modeling_t5 import T5ForConditionalGeneration
        encoder = T5ForConditionalGeneration.from_pretrained(init_model_path, config=plmconfig)
        
    else:
        print("fail to find your PLM <{}>".format(model_name))
        NotImplementedError
    
    if config.getint('distributed','local_rank') <= 0:
        if target_model :
            print("load target model = <{}> for distance compute.".format(model))
        else :
            print("load source model = <{}> for distance compute.".format(model))
            
    return encoder
    ##############
    

def get_params_for_prompt_optimization(module: torch.nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    return params


def init_optimizer(model, config):
    
    optimizer_type = config.get("train", "optimizer").lower()
    learning_rate = config.getfloat("train", "learning_rate")
    weight_decay = config.getfloat("train", "weight_decay")
    
    if config.getboolean("prompt", "prompt_tune"): #soft prompt tuning mode
        # will be trained only <encoder.{PLM}.embeddings.prompt_embeddings.weight>
        param_group = get_params_for_prompt_optimization(model)    
    
    else:#fine tuning mode
        param_group = model.parameters()

    if optimizer_type == "adam":
        optimizer = optim.Adam(param_group, lr=learning_rate,
                               weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(param_group, lr=learning_rate,
                              weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(param_group, lr=learning_rate,
                             weight_decay=weight_decay)
    else:
        raise NotImplementedError

    if config.getint('distributed','local_rank') <= 0:
        print("optimizer is setted - {}(lr={},wd = {})".format(optimizer_type,learning_rate,weight_decay))
        
    return optimizer

def init_optimizer_AE(model_AE,config):
    
    optimizer_type = config.get("train", "optimizer").lower()
    learning_rate = config.getfloat("train", "learning_rate")
    weight_decay = config.getfloat("train", "weight_decay")
    
    param_group = model_AE.parameters()
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(param_group, lr=learning_rate,
                               weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(param_group, lr=learning_rate,
                              weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(param_group, lr=learning_rate,
                             weight_decay=weight_decay)
    else:
        raise NotImplementedError

    if config.getint('distributed','local_rank') <= 0:
        print("optimizer_AE is setted - {}(lr={},wd = {})".format(optimizer_type,learning_rate,weight_decay))
        
    return optimizer    


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)

def init_output_function(config):
    
    from tools.output_tool import basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function
    
    output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "out1": output_function1,
    "acc": acc_output_function,
    "pearson": pearson_output_function}

    
    name = config.get("output", "output_function")
    if name in output_function_dic:
        return output_function_dic[name]
    else:
        print("check config('output','output_function') <= [Basic,Null,out1,acc,pearson]")
        raise NotImplementedError
