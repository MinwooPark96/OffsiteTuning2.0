#1e-3 distance mask lambda 0.1 pl loss

import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np

from tools.eval_tool import valid
from tools.init_tool import init_test_dataset, init_formatter
from src.reader.reader import init_dataset, init_formatter, init_test_dataset
import torch.nn as nn
import torch.optim as optim
#from model.optimizer import init_optimizer #저자주석
import transformers
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer, AE_1_layer_mutiple_100_paper,AE_transformer_layer,AE_1_layer_tokenwise,AE_auto_layer

from src.model.modelling_roberta import RobertaEmbeddings
from src.model.modelling_bert import BertEmbeddings
from src.model.modeling_t5 import T5EncoderModel
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer

from utils import utils

#minwoo
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# for save projector weight

    
def train(parameters, config, gpu_list, do_test=False, local_rank=-1, **params):
    
    epoch = config.getint("train", "epoch")
    
    train_valid_info = defaultdict(dict) 

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    
    if os.path.exists(output_path):
        if local_rank <= 0 :
            logger.warning("Output path exists, check whether need to change a name of model")
    
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    
    model_AE = utils.init_projector(config)

    optimizer_AE = parameters['optimizer'](model_AE,config)
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
    
    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_AE, step_size=step_size, gamma=gamma)
        
    exp_lr_scheduler.step(trained_epoch)
    source_encoder = utils.loadPLM(config,target_model=False).to("cuda")
    target_encoder = utils.loadPLM(config,target_model=True).to("cuda")
    
    # distance_function = torch.nn.MSELoss()
    # distance_function = torch.nn.PairwiseDistance()
    distance_function = torch.nn.CosineSimilarity(dim=2)
    lambda_ = config.getfloat('train','lambda')
    
    for epoch_num in range(trained_epoch, epoch):
        total_len = min([len(dataloader) for dataloader in parameters['train_dataset']])
        dataloader_list = parameters['train_dataset']
        
        for dataloader in dataloader_list:
            if len(dataloader_list) != 1:
                print(len(dataloader_list),dataloader_list)
                dataloader.sampler.set_epoch(epoch_num)
        
        dataloader_zipped = zip(*dataloader_list)
                
        if local_rank <=0:
            print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
        
        if total_len < 10000 and epoch_num==trained_epoch:
            more = "\t"

        start_time = timer()
        current_epoch = epoch_num
        
        model.eval() #model 부분은 건드리지 않기로함.
        
        
        acc_result = None
        
        performance = 0
        
        
        MTLoss = 0
        totalMTLoss = 0
        
        lossList = len(parameters['train_dataset'])*[0]
        totallossList = len(parameters['train_dataset'])*[0]
        
        distanceList = len(parameters['train_dataset'])*[0]
        totaldistanceList = len(parameters['train_dataset'])*[0]
        
        valid_total_loss = 0
        
        output_info = ""
        step = -1
        
        
       
        #각 batch 에 대하여 
        for step, dataloaders in enumerate(dataloader_zipped):
            # tensor to cuda
            for dataloader in dataloaders: 
                for dataset in dataloader:
                    for key in dataset.keys():
                        if isinstance(dataset[key], torch.Tensor):
                            if len(gpu_list) > 0:
                                dataset[key] = Variable(dataset[key].cuda())
                            else:
                                dataset[key] = Variable(dataset[key])
            
            model_AE.zero_grad() 
            
            if "T5" in config.get("target_model","model_base"):
                for idx,(source_dataset,target_dataset) in enumerate(dataloaders):
                    results = model(target_dataset, config, gpu_list, acc_result, "train", args=params, step=step, performance=performance, AE=model_AE)
                    loss, performance = results["loss"], results["performance"]
                    lossList[idx] = loss
                    
            else:
                for idx,(source_dataset,target_dataset) in enumerate(dataloaders):
                    results = model(target_dataset, config, gpu_list, acc_result, "train", AE=model_AE)
                    loss, acc_result = results["loss"], results["acc_result"]
                    
                    assert config.getboolean('projector','flatten') == False
                    
                    with torch.no_grad():
                        source_embedding = source_encoder.bert.embeddings(input_ids = source_dataset['inputx'][:,100:],attention_mask=source_dataset['mask'][:,100:])                        
                        
                        source_mask = source_dataset['mask'].unsqueeze(2).expand(source_dataset['inputx'].size(0),source_dataset['inputx'].size(1),source_embedding.size(2))[:,100:,:]
                        source_masked_embedding = source_mask * source_embedding
                        
                    
                        target_embedding = target_encoder.roberta.embeddings(input_ids = target_dataset['inputx'][:,100:],attention_mask=target_dataset['mask'][:,100:])
                        target_mask = target_dataset['mask'].unsqueeze(2).expand(target_dataset['inputx'].size(0),target_dataset['inputx'].size(1),target_embedding.size(2))[:,100:,:]
                        
                        target_masked_embedding = target_mask * target_embedding
                        
                    if config.getboolean('projector','flatten'):
                        source_module = source_masked_embedding @ torch.transpose(source_masked_embedding,1,2)
                        target_module = target_masked_embedding @ torch.transpose(target_masked_embedding,1,2)
                        distance = torch.norm(source_module-target_module,p='fro')*lambda_
                        # print(source_module.size())
                    
                    else :
                        source_module = model_AE(source_masked_embedding)
                        target_module = target_masked_embedding
                        
                        distance = distance_function(source_module,target_module)
                        distance = torch.mean(1-distance)*lambda_
                        
                        # distance = distance_loss(source_module,target_module)
                        # distance = torch.mean(distance)*lambda_
                        
                    lossList[idx] = loss    
                    distanceList[idx] = distance
                    
                    totallossList[idx] += loss
                    totaldistanceList[idx] += distance
                    
            MTLoss = sum(lossList)+ sum(distanceList)
            totalMTLoss += float(MTLoss)
            
            MTLoss.backward()
            optimizer_AE.step()

            if step % output_time == 0 and local_rank <= 0:
                if "T5" in config.get("target_model","model_base"):
                    delta_t = timer() - start_time
                    utils.output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        utils.gen_time_str(delta_t), utils.gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                 "%.3lf" % (totalMTLoss / (step + 1)), "\t", '\r', config)
                else:
                    output_info = output_function(acc_result, config)
                    delta_t = timer() - start_time
                    utils.output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        utils.gen_time_str(delta_t), utils.gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                 "%.3lf" % (totalMTLoss / (step + 1)), output_info, '\r', config)

            if "T5" in config.get("target_model","model_base") and int(step%10) == 0 and local_rank <=0 : 
                print("\t \t \t \t \t \t \t","Performance:", performance) #현재 여기에서 모든 local_rank 에서 performance 를 출력중이다. -> localrank 설정 추가

            global_step += 1

        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            if "T5" in config.get("target_model","model_base"):
                pass
            else:
                output_info = output_function(acc_result, config)
                #output_info_target = output_function(acc_result_target, config)
                delta_t = timer() - start_time
                utils.output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    utils.gen_time_str(delta_t), utils.gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                            "%.3lf" % (totalMTLoss / (step + 1)), output_info, None, config)
            
            utils.save_projector(os.path.join(output_path, "%d.pkl" % current_epoch),model_AE)
        
        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if current_epoch % test_time == 0 :
            with torch.no_grad():
                valid_epoch_loss_list = len(parameters['valid_dataset'])*[0]
                acc_result_eval_epoch = {'total': 0, 'right': 0}
                acc_result_eval_list = len(parameters['valid_dataset'])*[None]
                
                if "T5" in config.get("target_model","model_base"):
                    for idx,valid_dataset in enumerate(parameters['valid_dataset']):
                        acc_result_eval = valid(model, valid_dataset, current_epoch, config, gpu_list, output_function, AE=model_AE)
                        
                        acc_result_eval_list[idx] = acc_result_eval
                        
                        acc_result_eval_epoch['total'] += acc_result_eval['total']
                        acc_result_eval_epoch['right'] += acc_result_eval['right']

                else:
                    for idx,valid_dataset in enumerate(parameters['valid_dataset']):
                        
                        valid_loss, acc_result_eval = valid(model, valid_dataset, current_epoch, config, gpu_list, output_function, AE=model_AE)
                        
                        acc_result_eval_list[idx] = acc_result_eval
                        
                        acc_result_eval_epoch['total'] += acc_result_eval['total']
                        acc_result_eval_epoch['right'] += acc_result_eval['right']
                        
                        valid_epoch_loss_list[idx] = valid_loss

                    valid_total_loss += float(sum(valid_epoch_loss_list))
                    
                    
        if local_rank <=0 and not "T5" in config.get("target_model","model_base"):            
            
            if config.get("train","source_model"):
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + config.get("train","source_model")+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            
            elif config.get("train","prompt_emb"):
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + config.get("train","prompt_emb")+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            
            else :
                json_path = "result/" + config.get("data","train_dataset_type").replace(',','_') + '_' + 'NAN'+'_'+config.get("target_model","model_base") + config.get("target_model","model_size")
            
            if not os.path.exists('result'):
                os.mkdir('result')
            elif os.path.exists(json_path):
                with open(json_path,'r',encoding='utf-8') as file:
                    train_valid_info = json.load(file)
                    
            #each epoch sample loss
            train_valid_info["train_epoch_loss"][current_epoch] = round(float(sum(totallossList)) / (step + 1),6)
            train_valid_info["valid_epoch_loss"][current_epoch] = round(float(sum(valid_epoch_loss_list)) ,4)
            train_valid_info["distance_epoch_loss"][current_epoch] = round(float(sum(totaldistanceList)/ (step + 1)),6)
            
            
            #each epoch sample acc
            train_valid_info["train_epoch_acc"][current_epoch] = round(float(acc_result['right']/acc_result['total']),4)
            train_valid_info["valid_epoch_acc"][current_epoch] = round(float(acc_result_eval_epoch['right']/acc_result_eval_epoch['total']),4)
            
            train_data_list = config.get("data","train_dataset_type").split(',')
            valid_data_list = config.get("data","valid_dataset_type").split(',')
            
            
            for idx,data in enumerate(train_data_list):
                train_loss = data + "_train_loss"
                train_valid_info[train_loss][current_epoch] = round(float(totallossList[idx]/(step+1)),4)
                
            for idx,data in enumerate(valid_data_list):    
                valid_loss = data + "_valid_loss"
                valid_acc = data + "_valid_acc"
                train_valid_info[valid_loss][current_epoch] = round(float(valid_epoch_loss_list[idx]),4)
                train_valid_info[valid_acc][current_epoch] = round(float(acc_result_eval_list[idx]['right']/acc_result_eval_list[idx]['total']),4)
                
            with open(json_path,'w',encoding='utf-8') as make_file:
                json.dump(train_valid_info,make_file,indent = "\t")
        
        torch.distributed.barrier()
        exp_lr_scheduler.step(current_epoch)


