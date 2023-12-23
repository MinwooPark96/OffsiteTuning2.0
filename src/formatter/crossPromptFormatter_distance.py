#for distance
from transformers import AutoTokenizer,T5TokenizerFast
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class crossPromptFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        self.prompt_num = config.getint("prompt", "prompt_num")
        ##########
        
        self.target_len = config.getint("train", "target_len")
        
        if mode == 'train':
            
            self.source_model_name = config.get('train','source_model').lower()
            if "roberta" in self.source_model_name: 
                try:
                    self.source_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                except:
                    self.source_tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")
            
            elif "bert" in self.source_model_name:
                self.source_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            elif "t5" in self.source_model_name :
                
                try:
                    self.source_tokenizer = T5TokenizerFast.from_pretrained("t5-base")
                except:
                    self.source_tokenizer = T5TokenizerFast.from_pretrained("T5ForMaskedLM/t5-base")
            
            else:
                print("Have no matching in the formatter: formatter/crossPromptFormatter.py")
                exit()
            
        self.target_model_name = config.get("target_model","model_base").lower()
        
        
        if "roberta" in self.target_model_name: 
            try:
                self.target_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            except:
                self.target_tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")
        
        elif "bert" in self.model_name:
            self.target_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        elif "t5" in self.model_name :
            
            try:
                self.target_tokenizer = T5TokenizerFast.from_pretrained("t5-base")
            except:
                self.target_tokenizer = T5TokenizerFast.from_pretrained("T5ForMaskedLM/t5-base")
        
        else:
            print("Have no matching in the formatter: formatter/crossPromptFormatter.py")
            exit()
        
        
        self.prompt_prefix = [- (i + 1) for i in range(self.prompt_len)]
    
    def process_nonT5(self,data,mode,tokenizer):
        inputx = []
        mask = []
        label = []
        max_len = self.max_len + 3 + self.prompt_num 
        
        for ins in data:  
            sent1 = tokenizer.encode(ins["sent1"], add_special_tokens = False)
            try:
                sent2 = tokenizer.encode(ins["sent2"], add_special_tokens=False)
                tokens = self.prompt_prefix + [tokenizer.cls_token_id] + sent1 + [tokenizer.sep_token_id] + sent2 + [tokenizer.sep_token_id]
            
            except:
                tokens = self.prompt_prefix + [tokenizer.cls_token_id] + sent1 + [tokenizer.sep_token_id]

            if len(tokens) > max_len:
                tokens = tokens[:max_len - 1]
                tokens = tokens + [tokenizer.sep_token_id]
            
            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            
            if mode != "test":
                label.append(ins["label"])
            inputx.append(tokens)
        
        
        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "name" : ins['dataset']
        }

        return ret

    def process_T5(self,data,mode,tokenizer):
        inputx = []
        mask = []
        label = []
        max_len = self.max_len + 2 + self.prompt_num 
        data_name = data[0]["dataset"]
        
        
        if data_name.lower() == 'mnli' :
            dict_ = {0:"contradiction",1:"neutral",2:"entailment"}
        elif data_name.lower() == 'imdb':
            dict_ = dict_ = {0:"negative", 1:"positive"}
        elif data_name.lower() == 'laptop':
            dict_ = {0:"negative", 1:"neutral", 2:"positive", 3:"conflict"}            
        elif data_name.lower() == 'movierationales':
            dict_ = {0:"negative", 1:"positive"}
        elif data_name.lower() == 'mrpc':
            dict_ = {0:"false", 1:"true"}
        elif data_name.lower() == 'qnli':
            dict_ = {0:"contradiction",1:"entailment"}
        elif data_name.lower() == 'qqp':
            dict_ = {0:"false", 1:"true"}
        elif data_name.lower() == 'restaurant':
            dict_ = {0:"negative", 1:"moderate", 2:"positive", 3:"conflict"} #moderate -> nutural
        elif data_name.lower() == 'snli':
            dict_ = {0:"contradiction",1:"neutral",2:"entailment"}
        elif 'ethics' in data_name.lower() :
            dict_ = {0:"unacceptable", 1:"acceptable"}
        elif data_name.lower() == 'sst2':
            dict_ = {0:"negative", 1:"positive"}
        elif data_name.lower() == 'tweetevalsentiment':
            dict_ = {0:"negative", 1:"moderate", 2:"positive"}
        else :
            dict_ = {0:"false", 1:"true"}
        
        for ins in data:
            
            if data_name.lower() == 'mnli' or data_name.lower() == 'qnli' or data_name.lower() == 'snli':
                sent1 = tokenizer.encode("hypothesis: " + ins["sent1"], add_special_tokens = False)
                sent2 = tokenizer.encode("premise: " + ins["sent2"], add_special_tokens = False)
                tokens = self.prompt_prefix + sent1 + tokenizer.encode("[SEP] sentence: ", add_special_tokens=False)  + sent2

            elif data_name.lower() == 'mrpc' or data_name.lower() == 'qqp':
                sent1 = tokenizer.encode(ins["sent1"], add_special_tokens = False)
                try:
                    sent2 = tokenizer.encode(ins["sent2"], add_special_tokens=False)
                    tokens = self.prompt_prefix + sent1 + tokenizer.encode("<extra_id_0>", add_special_tokens=False)  + sent2

                except :
                    tokens = self.prompt_prefix + sent1

            elif data_name.lower() == 'ethicsdeontology':
                sent1 = tokenizer.encode(ins["sent1"], add_special_tokens = False)
                sent2 = tokenizer.encode(ins["sent2"], add_special_tokens=False)
                tokens = self.prompt_prefix + sent1 + sent2

            else :
                tokens = self.prompt_prefix + tokenizer.encode(ins["sent1"], add_special_tokens = False) 
                
            
            if len(tokens) >= max_len: #이전에 prefix 붙혀야함.
                tokens = tokens[:max_len-1]

            tokens = tokens + tokenizer.encode("</s>", add_special_tokens=False)
            
            tokens = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))

            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))

            target = tokenizer.encode(dict_[ins["label"]], add_special_tokens=False)
            
            if len(target) >= self.target_len:
                target = target[:self.target_len]
            
            target = target + [-100] * (self.target_len - len(target))

            if mode != "test":
                #label.append(target)
                label.append(target)
            inputx.append(tokens)


        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "name" : ins['dataset']
        }
        return ret

        
    def process(self, data, config, mode, *args, **params):
        
        local_rank = config.getint("distributed","local_rank")
        
        if mode == "train":
            if "t5" in self.source_model_name:
                dataset_source =  self.process_T5(data,mode,self.source_tokenizer)
            else :
                dataset_source =  self.process_nonT5(data,mode,self.source_tokenizer)
            
            # if local_rank <= 0 :
            #     print("[train] dataset_soure is setted from  <{}> , data = <{}>".format(self.model_name,data[0]['dataset']))
            
            if "t5" in self.target_model_name:
                dataset_target =  self.process_T5(data,mode,self.target_tokenizer)
            else :
                dataset_target =  self.process_nonT5(data,mode,self.target_tokenizer)
            
            # if local_rank <= 0 :
            #     print("[train] dataset_soure is setted from  <{}> , data = <{}>".format(self.model_name,data[0]['dataset']))
            
            
            return dataset_source,dataset_target
            
        else :
            
            if "t5" in self.target_model_name:
            
                return self.process_T5(data,mode,self.target_tokenizer)
            
            else :
                return self.process_nonT5(data,mode,self.target_tokenizer)
        
