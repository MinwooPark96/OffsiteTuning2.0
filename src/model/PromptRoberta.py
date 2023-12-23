import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os
import datasets

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")


class PromptRoberta(nn.Module):
    
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptRoberta, self).__init__()

        #Roberta variant sizes model: https://huggingface.co/readerbench/RoBERT-small
        #Model	       Weights	L	H	A	MLM accuracy	NSP accuracy
        #RoBERT-small	19M	    12	256	8	0.5363	        0.9687
        #RoBERT-base	114M	12	768	12	0.6511	        0.9802
        #RoBERT-large	341M	24	1024 24	0.6929	        0.9843


        #try:(저자 #)
        
        #[model] #model parameters
        # model_base = Roberta
        # model_name = PromptRoberta
        # model_size = base

        if config.get("target_model","model_size").lower()=="large":
            model = "roberta-large"
            ckp = "PLM/RobertaLargeForMaskedLM"
            self.hidden_size = 1024
        elif config.get("target_model","model_size").lower()=="small":
            model = "roberta-small"
            ckp = "PLM/RobertaSmallForMaskedLM"
            self.hidden_size = 256
        else: #현재 진행중
            model = "roberta-base"
            ckp = "PLM/RobertaForMaskedLM"
            self.hidden_size = 768
        '''
        except:
            model = "roberta-base"
            ckp = "RobertaForMaskedLM"
            self.hidden_size = 768
        '''

        if model == "roberta-small":
            self.plmconfig = AutoConfig.from_pretrained("RobertaSmallForMaskedLM/"+model)
            
        else:
            self.plmconfig = AutoConfig.from_pretrained(model)

        # [prompt]
        # prompt_tune = True
        # prompt_len = 100
        # prompt_num = 100

        #prompt_num 설정
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num") 
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_len") 

        if config.get("target_model","model_size").lower()=="large":
            self.init_model_path = str(ckp)+"/"+"PromptRobertaLarge_init_params"
        else:
            self.init_model_path = str(ckp)+"/"+"PromptRoberta_init_params"
        
        #저장파일이 존재한다면, 가져와야지.
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        #존재하지 않는다면 불러와야지.
        else:
            if model == "roberta-small":
                self.encoder = RobertaForMaskedLM.from_pretrained("RobertaSmallForMaskedLM/"+model, config=self.plmconfig)
            else:
                self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
            
            #불러왔으면 다음엔 가져오게 저장해야지.
            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")
            
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)

        ###########<본격적인 학습 시작부분>#################    
        

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output=False, **kwargs):
        data_name = data["name"].lower()
        
        if data_name == "stsb":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()        
        
        if prompt_emb_output == True:
            output, prompt_emb = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len)
        
        else:
            # print(data['inputx']) -> [-1,-2,...]
            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'])
            # print(self.encoder.get_input_embeddings())

        logits = output["logits"] # batch, seq_len, vocab_size #torch.Size([16, 231, 50265])

        mask_logits = logits[:, 0] # batch, vocab_size #torch.Size([16, 50265])



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
            print("PromptRoberta: What is this task? : PromptRoberta.py")
            #mask_logits:torch.Size([16, 50265])
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)


        # if data_name == "STSB":
        #     self.criterion = nn.MSELoss()
        # else:
        #     self.criterion = nn.CrossEntropyLoss()

        loss = self.criterion(score, data["label"])
        
        if data_name == "stsb":
            acc_result = pearson(score, data['label'], acc_result)
        else:
            acc_result = acc(score, data['label'], acc_result)

        if prompt_emb_output == True:
            return {'loss': loss, 'acc_result': acc_result}, prompt_emb, data['label']
        else:
            return {'loss': loss, 'acc_result': acc_result}

 

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
