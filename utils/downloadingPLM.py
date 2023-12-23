from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
import os
import torch

def downloadPLM(model_name,config):
    
    os.makedirs('PLM',exist_ok=True)
    
    os.makedirs('PLM/BertMediumForMaskedLM',exist_ok=True)
    os.makedirs('PLM/BertForMaskedLM',exist_ok=True)
    os.makedirs('PLM/BertLargeForMaskedLM',exist_ok=True)
    
    os.makedirs('PLM/RobertaForMaskedLM',exist_ok=True)
    os.makedirs('PLM/RobertaLargeForMaskedLM',exist_ok=True)
    
    os.makedirs('PLM/T5SmallForMaskedLM',exist_ok=True)
    os.makedirs('PLM/T5ForMaskedLM',exist_ok=True)
    os.makedirs('PLM/T5LargeForMaskedLM',exist_ok=True)
    os.makedirs('PLM/T53BForMaskedLM',exist_ok=True)
    os.makedirs('PLM/T511BForMaskedLM',exist_ok=True)
    
    model_name = model_name.lower()
    
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
        print("Check your model name - utils.downlaodingPLM.py <{}>".format(model_name))
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

    if os.path.exists(init_model_path+"/pytorch_model.bin"):
        
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
            exit()
    else:
        if "roberta" in model_name:
            from src.model.modelling_roberta import RobertaForMaskedLM
            encoder = RobertaForMaskedLM.from_pretrained(model, config=plmconfig)
            os.makedirs(init_model_path,exist_ok=True)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            encoder = RobertaForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
            
        elif "bert" in model_name:
            from src.model.modelling_bert import BertForMaskedLM
            encoder = BertForMaskedLM.from_pretrained(model, config=plmconfig)
            os.makedirs(init_model_path,exist_ok=True)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            encoder = BertForMaskedLM.from_pretrained(init_model_path, config=plmconfig)
        
        elif 't5' in model_name :
            from src.model.modeling_t5 import T5ForConditionalGeneration
            encoder = T5ForConditionalGeneration.from_pretrained(model, config=plmconfig)

            os.makedirs(init_model_path,exist_ok=True)
            torch.save(encoder.state_dict(), str(init_model_path)+"/pytorch_model.bin")
            encoder = T5ForConditionalGeneration.from_pretrained(init_model_path, config=plmconfig)

        else:
            exit()
    
    if config.getint('distributed','local_rank') <= 0:
        print("download model = <{}>".format(model))
            
    return encoder
    ##############