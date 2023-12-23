import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from transformers import AutoConfig
from .modeling_t5 import T5ForConditionalGeneration
from torchnlp.metrics import get_moses_multi_bleu

# from transformers import T5TokenizerFast
# tokenizer = T5TokenizerFast.from_pretrained("T5ForMaskedLM/t5-base")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-base")

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoother = SmoothingFunction()

class crossPromptT5(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(crossPromptT5, self).__init__()

        try:
            if config.get("target_model","model_size").lower()=="small":
                model = "t5-small"
                ckp = "PLM/T5SmallForMaskedLM"
                self.hidden_size = 512
            elif config.get("target_model","model_size").lower()=="large":
                model = "t5-large"
                ckp = "PLM/T5LargeForMaskedLM"
                self.hidden_size = 1024
            elif config.get("target_model","model_size").lower()=="b3":
                model = "t5-b3"
                ckp = "PLM/T5B3ForMaskedLM"
                self.hidden_size = 1024
            
            else:
                model = "t5-base"
                ckp = "PLM/T5ForMaskedLM"
                self.hidden_size = 768
        except:
            model = "t5-base"
            ckp = "PLM/T5ForMaskedLM"
            self.hidden_size = 768


        self.plmconfig = AutoConfig.from_pretrained(model)
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")

        
        if config.get("train","prompt_emb"):
            if config.get('distributed','local_rank') <= 0:     
                print("load source prompt from <{}> in crossPrompt.py".format("task_prompt_emb/"+config.get("train","prompt_emb")+"/task_prompt"))
            filename = "task_prompt_emb/"+config.get("train","prompt_emb")+"/task_prompt"
            self.task_specific_prompt_emb = torch.load(filename).to('cuda')
            # self.task_specific_prompt_emb = torch.unsqueeze(self.task_specific_prompt_emb,0)
        
        else :
            if config.get('distributed','local_rank') <= 0:
                print("do not load any source prompt in crossPromptT5.py")
        
        self.init_model_path = str(ckp)+"/"+"PromptT5_init_params"
        
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            #self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            self.encoder = T5ForConditionalGeneration.from_pretrained(model, config=self.plmconfig)

            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")

            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)

    def forward(self, data, config, gpu_list, acc_result, mode, **kwargs):

        data_name = data["name"]
        
        if not config.get('train','prompt_emb') and config.get('train','source_model'):

            filename = "task_prompt_emb/" + data_name + "Prompt" + config.get("train","source_model")+"/task_prompt"    
            self.task_specific_prompt_emb = torch.load(filename).to('cuda') #[100,768]
            # print(filename)
            # self.task_specific_prompt_emb = torch.unsqueeze(self.task_specific_prompt_emb,0)
        
        model_AE = kwargs["AE"]
        
        target_prompt = model_AE(self.task_specific_prompt_emb.reshape(1,int(self.task_specific_prompt_emb.shape[0])*int(self.task_specific_prompt_emb.shape[1])))
        target_prompt = target_prompt.reshape(int(target_prompt.shape[1]))
        target_prompt = target_prompt.reshape(int(self.encoder.prompt_embeddings.weight.data.shape[0]),int(self.encoder.prompt_embeddings.weight.data.shape[1]))
        
        # task_specific_prompt_emb = torch.unsqueeze(self.task_specific_prompt_emb,0)
        # task_specific_prompt_emb_ = task_specific_prompt_emb.reshape( int(task_specific_prompt_emb.shape[0]), int(task_specific_prompt_emb.shape[1])*int(task_specific_prompt_emb.shape[2]))
        # task_specific_prompt_emb_ = model_AE(task_specific_prompt_emb_)
        # target_prompt = task_specific_prompt_emb_.reshape(int(task_specific_prompt_emb_.shape[1]))
        # target_prompt = target_prompt.reshape(int(self.encoder.prompt_embeddings.weight.data.shape[0]),int(self.encoder.prompt_embeddings.weight.data.shape[1]))
        
        
        self.encoder.prompt_embeddings.weight.data = target_prompt
        self.encoder.encoder.prompt_tokens.weight.data = target_prompt
        self.encoder.decoder.prompt_tokens.weight.data = target_prompt
        
        if mode == 'train':
            inputx = []
            mask = []
            labels = []

            output = self.encoder(input_ids=data["inputx"], labels=data["label"])
            performance = kwargs["performance"]

            if int(kwargs["step"]%10) == 0:
                # print(data["inputx"][0])
                # print(data['label'])
                gen = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))
                # print(gen)
                
                if "squad" in data_name or "nq_open" in data_name or "multi_news" in data_name or "samsum" in data_name:
                    performance = train_bleu(gen['sequences'], data["label"], data_name)

                else:
                    performance = train_acc(gen['sequences'], data["label"], data_name)
                            
            return {'loss': output["loss"], 'performance':performance}


        elif mode == 'valid':

            model_AE = kwargs["AE"]
        
        
            output = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))
            if "squad" in data_name or "nq_open" in data_name or "multi_news" in data_name or "samsum" in data_name:
                acc_result = bleu(output['sequences'], data["label"], acc_result, data_name)

            else:
                hidden_score = output["scores"][0]
                acc_result = acc(output['sequences'], data["label"], acc_result, data_name, hidden_score=hidden_score)

            return {'acc_result':acc_result}





def train_acc(score, label, dataset):
    score = score[:,1:2]
    label = label[:,0:1]

    total = int(label.shape[0])
    right = int((score == label).int().sum())
    
    acc_result = round(float(right/total),4)
    # print(score[0],label[0],acc_result,"train_acc")
    return acc_result



def acc(score, label, acc_result, dataset, hidden_score=None):

    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}


    acc_result['total'] += int(label.shape[0])
    #print(label)
    #exit()
    label = label[:,0:1]
    
    if dataset.lower() == "imdb":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "sst2":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0 
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "laptop":
        #negative: 2841, moderate:8107, positive:1465, conflict:4129
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1), hidden_score[:,4129].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        label[label==4129] = 3
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "restaurant": #neutral 이 맞는듯
        #negative: 2841, moderate:8107, positive:1465, conflict:4129
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1), hidden_score[:,4129].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        label[label==4129] = 3
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "movierationales":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "tweetevalsentiment":
        #negative: 2841, moderate:8107, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "mnli":
        #contradiction: 27252, neutral: 7163, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,7163].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==7163] = 1
        label[label==3] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "qnli":
        #contradiction: 27252, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==3] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset.lower() == "snli":
        #contradiction: 27252, neutral: 7163, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,7163].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==7163] = 1
        label[label==3] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "ethics" in dataset: #### process_T5 에서 안함
        #unacceptable: 29452, acceptable: 9961
        score = torch.cat([hidden_score[:,29452].unsqueeze(1), hidden_score[:,9961].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==29452] = 0
        label[label==9961] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "qqp" in dataset.lower() or "mrpc" in dataset.lower():
        #false: 6136, true:1176
        score = torch.cat([hidden_score[:,6136].unsqueeze(1), hidden_score[:,1176].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==6136] = 0
        label[label==1176] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "activate" in dataset.lower(0): #process_T5 에서 안함
        #false: 6136, true:1176
        score = torch.cat([hidden_score[:,6136].unsqueeze(1), hidden_score[:,1176].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==6136] = 0
        label[label==1176] = 1
        label = label.reshape(int(label.shape[0]))
    else:
        print("Eval metrics wrong!!!..crossPromptT5.py/acc")
        exit()

    #acc_result['total'] += int(label.shape[0])
    # print(score,label,"acc")
    acc_result['right'] += int((score == label).int().sum())

    return acc_result





def train_bleu(score, label, dataset):
    total_bleu = 0
    length = len(label)

    #references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    references = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    #hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    hypotheses = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]

    total_bleu = 0
    for l in range(len(hypotheses)):
        print(references[l])
        #y = [references[l].lower().split()]
        y = [references[l]]
        print(hypotheses[l])
        #y_ = hypotheses[l].lower().split()
        y_ = hypotheses[l]
        print("-----")
        if len(y)!=0 and len(y_)!=0:
            #b = sentence_bleu(y, y_, weights=(0.35, 0.35, 0.15, 0.15), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
            b = sentence_bleu(y, y_, weights=(0.7, 0.3, 0.0, 0.0), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
        else:
            b = 0
        print(b)
        total_bleu+=b
    print("========")
    result = round(float(total_bleu/length),4)

    return result




def bleu(score, label, acc_result, dataset):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    acc_result['total'] += int(label.shape[0])

    #references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    references = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    #hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    hypotheses = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]

    total_bleu = 0
    for l in range(len(hypotheses)):
        #print(references[l].lower().split())
        #y = [references[l].lower().split()]
        y = [references[l]]
        #print(hypotheses[l].lower().split())
        #y_ = hypotheses[l].lower().split()
        y_ = hypotheses[l]
        #print("-----")
        if len(y)!=0 and len(y_)!=0:
            b = sentence_bleu(y, y_, weights=(0.7, 0.3, 0.0, 0.0), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
            #b = sentence_bleu(y, y_, weights=(0.35, 0.35, 0.15, 0.15), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
        else:
            b = 0
        #print(b)
        total_bleu+=b
    #print("========")

    '''
    references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    total_bleu = get_moses_multi_bleu(hypotheses, references, lowercase=True)
    if total_bleu == None:
        total_bleu = 0
    '''
    acc_result['right'] += int(total_bleu)

    return acc_result



