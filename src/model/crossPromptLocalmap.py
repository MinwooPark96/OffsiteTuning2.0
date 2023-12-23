import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os
import datasets

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM

#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")

import logging

logger = logging.getLogger(__name__)



#class crossPromptRoberta(nn.Module):
class crossPrompt(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
    #model(data, config, gpu_list, acc_result, "train", AE=model_AE) from train_tool_cross.py
    
        super(crossPrompt, self).__init__()
        local_map = config.getboolean("train","local_map")
        assert local_map == True
        
        if "roberta" in config.get("target_model","model_base").lower():
            try:
                if config.get("target_model","model_size").lower()=="large":
                    model = "roberta-large"
                    ckp = "PLM/RobertaLargeForMaskedLM"
                    self.hidden_size = 1024
                else:
                    model = "roberta-base"
                    ckp = "PLM/RobertaForMaskedLM"
                    self.hidden_size = 768
                
            except:
                model = "roberta-base"
                ckp = "PLM/RobertaForMaskedLM"
                self.hidden_size = 768
        
        elif "bert" in config.get("target_model","model_base").lower():
            try:
                if config.get("target_model","model_size").lower()=="large":
                    model = "bert-large"
                    ckp = "PLM/BertLargeForMaskedLM"
                    self.hidden_size = 1024
                elif config.get("target_model","model_size").lower()=="base":
                    model = "bert-base-uncased"
                    ckp = "PLM/BertForMaskedLM"
                    self.hidden_size = 768
                elif config.get("target_model","model_size").lower()=="medium":
                    model = "prajjwal1/bert-medium"
                    ckp = "PLM/BertMediumForMaskedLM"
                    self.hidden_size = 512
            except:
                model = "bert-base-uncased"
                ckp = "PLM/BertForMaskedLM"
                self.hidden_size = 768
        else:
            print("Wrong!!!")
            print("crossPrompt.py Error")
            exit()
        
        if config.get("train","prompt_emb"):
            if config.get('distributed','local_rank') <= 0:     
                print("load source prompt from <{}> in crossPrompt.py".format("task_prompt_emb/"+config.get("train","prompt_emb")+"/task_prompt"))
            filename = "task_prompt_emb/"+config.get("train","prompt_emb")+"/task_prompt"
            
            self.task_specific_prompt_emb = torch.load(filename).to('cuda')
            self.task_specific_prompt_emb = torch.unsqueeze(self.task_specific_prompt_emb,0)                                                        
            
        else :
            if config.get('distributed','local_rank') <= 0:
                print("will load source prompt <{}>in crossPrompt.py".format(config.get('train','source_model')))
            
        
        self.plmconfig = AutoConfig.from_pretrained(model)
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")

        if "bert-medium" in model: #"prajjwal1/bert-medium"
            model = "bert-medium"

        if config.get("target_model","model_size").lower()=="large":
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Large"+"_init_params"
        
        elif config.get("target_model","model_size").lower()=="medium":
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Medium"+"_init_params"
        
        else:
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"_init_params"

        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            if "roberta" in config.get("target_model","model_base").lower():
                from .modelling_roberta import RobertaForMaskedLM
                self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            elif "bert" in config.get("target_model","model_base").lower():
                from .modelling_bert import BertForMaskedLM
                self.encoder = BertForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            else:
                print("Wrong/crossPrompt.py")
                exit()
        else:
            if "roberta" in config.get("target_model","model_base").lower():
                from .modelling_roberta import RobertaForMaskedLM
                self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
                os.mkdir(self.init_model_path)
                torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
                print("Save Done")
                self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            elif "bert" in config.get("target_model","model_base").lower():
                from .modelling_bert import BertForMaskedLM
                self.encoder = BertForMaskedLM.from_pretrained(model, config=self.plmconfig)
                os.mkdir(self.init_model_path)
                torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
                print("Save Done")
                self.encoder = BertForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            else:
                print("Wrong/crossPrompt.py")
                exit()
        ##############
        

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output="replace_task_specific_prompt_emb", **kwargs):

        
        data_name = data["name"]
        if config.get('train','source_model'):
            filename = "task_prompt_emb/" + data_name + "Prompt" + config.get("train","source_model")+"/task_prompt"    
            self.task_specific_prompt_emb = torch.load(filename).to('cuda')
            self.task_specific_prompt_emb = torch.unsqueeze(self.task_specific_prompt_emb,0)                                                        
        
        # print(filename)
        # print(data['inputx'][0])
        
        # if config.get("distributed",'local_rank')<=0:
        #     print("crossPrompt.py is running.. with data = <{}>".format(data_name))
        
        # print(data_name,filename)
        
        if data_name.lower() == "stsb":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
        if mode == 'train' :
            batch_size = config.getint('train',"batch_size")
        else :
            batch_size = config.getint('eval',"batch_size")
            
        task_specific_prompt_emb = self.task_specific_prompt_emb.repeat(batch_size,1,1) 
        model_AE = kwargs["AE"]
        
        task_specific_prompt_emb_ = task_specific_prompt_emb.reshape( int(task_specific_prompt_emb.shape[0]), int(task_specific_prompt_emb.shape[1])*int(task_specific_prompt_emb.shape[2]))
        task_specific_prompt_emb_ = model_AE(task_specific_prompt_emb_)
        dim_out = int(int(model_AE.decoder.weight.shape[0])/int(task_specific_prompt_emb.shape[1]))
        task_specific_prompt_emb = task_specific_prompt_emb_.reshape(int(task_specific_prompt_emb.shape[0]),int(task_specific_prompt_emb.shape[1]),dim_out)
                    
        output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len, task_specific_prompt_emb=task_specific_prompt_emb)
        # self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)


        logits = output["logits"] # batch, seq_len, vocab_size #torch.Size([16, 231, 50265])
        
        mask_logits = logits[:, 0] # batch, vocab_size #torch.Size([16, 50265])

        if config.get("target_model","model_base").lower() == "roberta":
            if data_name == "laptop" or data_name == "restaurant":
                #sentiment
                #mo_dict={"positive":22173,"moderate":19397,"negative":33407,"conflict":17075}
                score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1), mask_logits[:,17075].unsqueeze(1)], dim=1)

            elif data_name == "tweetevalsentiment":
                #sentiment
                #mo_dict={"positive":22173,"moderate":19397,"negative":33407}
                score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1)], dim=1)

            elif data_name == "sst2" or data_name == "imdb" or data_name == "movierationales":
                #sentiment
                #mo_dict={"positive":22173,"negative":33407}
                score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:,22173].unsqueeze(1)], dim=1)
                
            elif data_name == "mnli" or data_name == "snli" or data_name == "anli":
                #NLI
                #mo_dict={"yes":10932,"neutral":12516,"no":2362}
                score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 12516].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
            elif data_name == "rte" or "recast" in data_name:
                #NLI
                #mo_dict={"yes":10932,"no":2362}
                score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
            elif data_name == "wnli":
                #NLI
                #mo_dict={"yes":10932,"no":2362}
                score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
            elif data_name == "qnli":
                #NLI
                #mo_dict={"yes":10932,"no":2362}
                score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
            elif data_name == "mrpc":
                #paraphrase
                #mo_dict={"true":29225,"false":22303}
                score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
            elif data_name == "qqp":
                #paraphrase
                #mo_dict={"true":29225,"false":22303}
                score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
            elif data_name == "stsb":
                score = mask_logits[:, 1032]
            elif data_name == "emobankarousal" or data_name == "persuasivenessrelevance" or data_name == "persuasivenessspecificity" or data_name == "emobankdominance" or data_name == "squinkyimplicature" or data_name == "squinkyformality":
                #(["high”]): 3530
                #(["low”]): 5481
                score = torch.cat([mask_logits[:,5481].unsqueeze(1), mask_logits[:,3530].unsqueeze(1)], dim=1)
            elif "ethics" in data_name:
                #"acceptable":[32047], "un":[879]
                score = torch.cat([mask_logits[:, 897].unsqueeze(1), mask_logits[:,32047].unsqueeze(1)], dim=1)
            else:
                #Other
                print("crossPrompt.py - roberta: What is this task? : PromptRoberta.py")
                #mask_logits:torch.Size([16, 50265])
                #mo_dict={"yes":10932,"no":2362}
                score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)

        elif config.get("target_model","model_base").lower() == "bert":
            if data_name == "laptop" or data_name == "restaurant":
                #sentiment
                #mo_dict={"positive":3893,"moderate":8777,"negative":4997,"conflict":4736}
                score = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:, 8777].unsqueeze(1), mask_logits[:, 3893].unsqueeze(1), mask_logits[:,4736].unsqueeze(1)], dim=1)
            elif data_name == "tweetevalsentiment":
                #mo_dict={"positive":3893,"moderate":8777,"negative":4997}
                score = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:, 8777].unsqueeze(1), mask_logits[:, 3893].unsqueeze(1)], dim=1)

            elif data_name == "sst2" or data_name == "imdb" or data_name == "movierationales":
                #sentiment
                #mo_dict={"positive":3893,"negative":4997}
                score = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:,3893].unsqueeze(1)], dim=1)
            elif data_name == "mnli" or data_name == "snli" or data_name == "anli" or data_name == "recastfactuality":
                #NLI
                #mo_dict={"yes":2748,"neutral":8699,"no":2053}
                score = torch.cat([mask_logits[:, 2053].unsqueeze(1), mask_logits[:, 8699].unsqueeze(1), mask_logits[:, 2748].unsqueeze(1)], dim=1)
            elif data_name == "rte":
                #NLI
                #mo_dict={"yes":2748,"no":2053}
                score = torch.cat([mask_logits[:, 2053].unsqueeze(1), mask_logits[:, 2748].unsqueeze(1)], dim=1)
            elif data_name == "wnli":
                #NLI
                #mo_dict={"yes":2748,"no":2053}
                score = torch.cat([mask_logits[:, 2053].unsqueeze(1), mask_logits[:, 2748].unsqueeze(1)], dim=1)
            elif data_name == "qnli" or "recast" in data_name:
                #NLI
                #mo_dict={"yes":2748,"no":2053}
                score = torch.cat([mask_logits[:, 2053].unsqueeze(1), mask_logits[:, 2748].unsqueeze(1)], dim=1)
            elif data_name == "mrpc":
                #paraphrase
                #mo_dict={"true":2995,"false":6270}
                score = torch.cat([mask_logits[:, 6270].unsqueeze(1), mask_logits[:,2995].unsqueeze(1)], dim=1)
            elif data_name == "qqp":
                #paraphrase
                #mo_dict={"true":2995,"false":6270}
                score = torch.cat([mask_logits[:, 6270].unsqueeze(1), mask_logits[:,2995].unsqueeze(1)], dim=1)
            elif data_name == "stsb":
                score = mask_logits[:, 10932]
            elif data_name == "emobankarousal" or data_name == "persuasivenessrelevance" or data_name == "persuasivenessspecificity" or data_name == "emobankdominance" or data_name == "squinkyimplicature" or data_name == "squinkyformality":
                #"low" [2659]
                #"high" [2152]
                score = torch.cat([mask_logits[:, 2659].unsqueeze(1), mask_logits[:, 2152].unsqueeze(1)], dim=1)
            elif "ethics"in data_name:
                #21873:unacceptable,  11701:acceptable

                score = torch.cat([mask_logits[:, 21873].unsqueeze(1), mask_logits[:, 11701].unsqueeze(1)], dim=1)
            else:
                print("crossPrompt - bert : What is this task?")
                #Other
                #mask_logits:torch.Size([16, 50265])
                #mo_dict={"yes":10932,"no":2362}
                #mo_dict={"yes":2748,"no":2053}
                score = torch.cat([mask_logits[:, 2053].unsqueeze(1), mask_logits[:, 2748].unsqueeze(1)], dim=1)

        else:
            print("Cannot access. model/crossPrompt.py - do not compute score. model fails to identify which model do you select")
            exit()

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        return {'loss': loss, 'acc_result': acc_result}



def acc_mlm(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}

    predict = torch.max(score, dim = 2)[1]

    NOT_MASK = [label!=-100]
    predict = predict[NOT_MASK]
    label = label[NOT_MASK]

    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())

    return acc_result


def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())

    return acc_result


def pearson(score, label, acc_result):
    stsb_result = cal_pearson(score, label)
    if acc_result is None:
        acc_result = {'total_pearson': 0, 'batch_num': 0}
    acc_result['total_pearson'] += stsb_result['pearson']
    acc_result['batch_num'] += 1
    return acc_result


def cal_pearson(score, label):
    tmp_result = {}
    score_bar = torch.mean(score, dim=-1)
    label_bar = torch.mean(label, dim=-1)
    numerator = torch.sum(torch.mul(score-score_bar, label - label_bar), dim=-1)
    denominator = torch.sqrt(torch.sum((score-score_bar) ** 2, dim=-1)) * torch.sqrt(torch.sum((label-label_bar) ** 2, dim=-1))
    pearson_result = numerator / denominator
    tmp_result['pearson'] = pearson_result.item()
    return tmp_result
