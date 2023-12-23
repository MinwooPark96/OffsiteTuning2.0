import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import random

class crossDataset(Dataset):
    
    def __init__(self, config, mode, data_name, encoding="utf8", *args, **params):

        self.config = config
        self.mode = mode
        
        self.dataset_name = data_name.lower()

        self.dataset, self.length = pre_processing(self.dataset_name, mode,self.config)
        
        random.shuffle(self.dataset)
        
        if config.getint("distributed",'local_rank')<=0 :
            print("data_name = {}, mode = {}, number of data = {}, local_map = always True".format(self.dataset_name,mode,len(self.dataset)))
    
    def __getitem__(self, item):
        #return self.all[item]
        return self.dataset[item]

    def __len__(self):
        #return len(self.data)
        return len(self.dataset)



def pre_processing(data_name, mode,config):

        if "stsb" == data_name:
            return pre_data_stsb(mode,config)
        if "sst2" == data_name:
            return pre_data_sst2(mode,config)
        if "restaurant" == data_name:
            return pre_data_restaurant(mode,config)
        if "qnli" == data_name:
            return pre_data_qnli(mode,config)
        if "qqp" == data_name:
            return pre_data_qqp(mode,config)
        if "mrpc" == data_name:
            return pre_data_mrpc(mode,config)
        if "wnli" == data_name:
            return pre_data_wnli(mode,config)
        if "rte" == data_name:
            return pre_data_rte(mode,config)
        if "mnli" == data_name:
            return pre_data_mnli(mode,config)
        if "laptop" == data_name:
            return pre_data_laptop(mode,config)
        if "imdb" == data_name:
            return pre_data_imdb(mode,config)
        if "snli" == data_name:
            return pre_data_snli(mode,config)
        if "anli" == data_name:
            return pre_data_anli(mode,config)
        if "tweetevalsentiment" == data_name:
            return pre_data_tweetevalsentiment(mode,config)
        if "movierationales" == data_name:
            return pre_data_movierationales(mode,config)
        if "ethicsdeontology" == data_name:
            return pre_data_ethicsdeontology(mode,config)
        if "ethicsjustice" == data_name:
            return pre_data_ethicsjustice(mode,config)
        if "sst-2_s1" == data_name:
            return pre_data_sst2(mode,config,"SST-2_s1")
        if "sst-2_s2" == data_name:
            return pre_data_sst2(mode,config,"SST-2_s2")
        if "imdb_s1" == data_name:
            return pre_data_imdb(mode,config,"IMDB_s1")
        if "imdb_s2" == data_name:
            return pre_data_imdb(mode,config,"IMDB_s1")

#label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict, 9:low, 10:high}




#data/WNLI/train.tsv
#map : 없음
def pre_data_wnli(mode,config):
    '''
    data = load_dataset('glue', 'wnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''
    #만약 t5 이거나 cross 가 안들어감 
    local_map = True
    
    if local_map :
        pass
    
    dataset_size = config.getint("train","dataset_size")
    
    tsv_file = open("data/WNLI/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/WNLI/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/WNLI/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"wnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": int(ins['label']), "dataset":"wnli"} for ins in validation_data]
    else:
        if dataset_size == -1:
            data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": int(ins['label']) , "dataset":"wnli"} for ins in train_data]
        else :
            data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": int(ins['label']) , "dataset":"wnli"} for ins in train_data[:dataset_size]]
    
    return data, len(data)


#data = load_dataset('glue','rte')
#map : 없음
def pre_data_rte(mode,config):
    
    local_map = True
    
    if local_map :
        pass
    '''
    if mode == "train":
        d = load_dataset('glue', 'rte')
    else:
        d = csv.reader(open("./data/RTE/test.tsv", "r"), delimiter='\t')
    '''
    data = load_dataset('glue','rte')
    '''
    if mode=='valid':
        mode = "validation"
    data = data[mode]
    '''
    
    dataset_size = config.getint("train","dataset_size")
    
    
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    '''
    tsv_file = open("data/RTE/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/RTE/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/RTE/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")
    '''


    dict_={0:1,1:0}
    #dict_={'not_entailment':0,'entailment':1}

    if mode == "test":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "dataset":"rte"} for ins in test_data]
    elif mode == "train":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in train_data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    elif mode == "valid":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in validation_data]

    return data, len(data)


#data = load_dataset('glue', 'qnli')
#if local map : {0:1,1:0} else : {0:1,1:0}
def pre_data_qnli(mode,config):
    
    local_map = True
    if local_map :
        dict_={0:1,1:0}
    else :
        dict_={0:1,1:0}
    
    dataset_size = config.getint("train","dataset_size")
    
    data = load_dataset('glue', 'qnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']


    if mode == "test":
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    '''
    tsv_file = open("data/QNLI/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")


    #dict_={0:1,1:0}
    dict_={"not_entailment":0,"entailment":1}

    #data=[]
    if mode == "test":
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]
    '''

    # print("Done")
    # print(mode, "the number of data", len(data))

    return data, len(data)


#data = load_dataset('glue', 'stsb')
#map 없음
def pre_data_stsb(mode,config):
    local_map = True
    if local_map:
        pass
    
    dataset_size = config.getint("train","dataset_size")
    
    data = load_dataset('glue', 'stsb')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"stsb"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in train_data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    return data, len(data)


#./data/SST-2/train.tsv
#if local map : {0:5,1:7} else : {0:0,1:1}
def pre_data_sst2(mode,config,data_name=None):
    
    local_map = True
    
    if not local_map :
        _map={0:5, 1:7}
    else :
        _map = {0:0,1:1}
            
    dataset_size = config.getint("train","dataset_size")
    
    
    if mode == "train":
        d = csv.reader(open("./data/SST-2/train.tsv", "r"), delimiter='\t', quotechar='"')
    elif mode == "valid" or mode == "validation":
        d = csv.reader(open("./data/SST-2/dev.tsv", "r"), delimiter='\t', quotechar='"')
    else:
        d = csv.reader(open("./data/SST-2/test.tsv", "r"), delimiter='\t', quotechar='"')

    data = [row for row in d]
    
    if mode == "test":
        data = [{"sent1": ins[0].strip(), "dataset":"sst2"} for ins in data[1:]]
    
    elif mode == "train":
        data = [{"sent1": ins[0].strip(), "label": _map[int(ins[1].strip())], "dataset":"sst2"} for ins in data[1:]]
        
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    else:
        data = [{"sent1": ins[0].strip(), "label": _map[int(ins[1].strip())], "dataset":"sst2"} for ins in data[1:]]

    return data, len(data)



#./data/restaurant/train.json"
# if local map : emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
# else : emo_dict={"positive":7,"netural":6,"negative":5,"conflict":8} 
def pre_data_restaurant(mode,config):
    local_map = True
    
    if not local_map:
        emo_dict={"positive":7,"netural":6,"negative":5,"conflict":8}
    else :
        emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}

    dataset_size = config.getint("train","dataset_size")
    
    if mode == "train":
        data = []
        with open("./data/restaurant/train.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif mode == "valid":
        data = []
        with open("./data/restaurant/test.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = []
        with open("./data/restaurant/test.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    if mode == "test":
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "dataset":"restaurant"})
    elif mode == 'valid':
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "label": emo_dict[ins['aspects']["polarity"][0]], "dataset":"restaurant"})
    else:
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "label": emo_dict[ins['aspects']["polarity"][0]], "dataset":"restaurant"})
        
        if dataset_size == -1 :
            pass
        else :
            self_data = self_data[:dataset_size]
        
    # print(mode, "the number of data", len(data))
    # print(data)

    return self_data, len(self_data)


#data/QQP/train.tsv
# if local map : {0:0,1:1} else {0:2,1:4} (fasle,true)
def pre_data_qqp(mode,config):
    
    local_map = True
    
    if not local_map :
        _map={0:2,1:4}
    else :
        _map={0:0,1:1}

    dataset_size = config.getint("train","dataset_size")
    
    # data = load_dataset('glue', 'qqp')
    # train_data = self.data['train']
    # validation_data = self.data['validation']
    # test_data = self.data['test']
    
    tsv_file = open("data/QQP/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QQP/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QQP/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")
    
    

    if mode == "test":
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question2'], "dataset":"qqp"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question2'].strip(), "label": _map[int(ins['is_duplicate'])], "dataset":"qqp"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question2'].strip(), "label": _map[int(ins['is_duplicate'])], "dataset":"qqp"} for ins in
                     train_data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    return data, len(data)

#data/MRPC/msr_paraphrase_train.txt
# if local map : {0:0,1:1} else {0:2,1:4} (fasle,true)
def pre_data_mrpc(mode,config):
    local_map = True
    
    if not local_map :
        _map={0:2,1:4}
    else :
        _map={0:0,1:1}

    
    dataset_size = config.getint("train","dataset_size")
    
    tsv_file = open("data/MRPC/msr_paraphrase_train.txt", encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MRPC/msr_paraphrase_test.txt", encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MRPC/msr_paraphrase_test.txt", encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    

    if mode == "test":
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "dataset":"mrpc"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "label": _map[int(ins['Quality'])], "dataset":"mrpc"} for ins in validation_data]
    else:
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "label": _map[int(ins['Quality'])], "dataset":"mrpc"} for ins in train_data]
        
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    return data, len(data)


# data = load_dataset('glue', 'mnli')
# if local_map : {0:2,1:1,2:0} else {2:0,1:3,0:1}
def pre_data_mnli(mode,config):
    
    local_map = True
    
    if not local_map :
        _dict={2:0,1:3,0:1}
    else :
        _dict = {0:2,1:1,2:0}
    
    dataset_size = config.getint("train","dataset_size")
    
    data = load_dataset('glue', 'mnli')
    train_data = data['train']
    
    validation_matched_data = data['validation_matched']
    validation_mismatched_data = data['validation_mismatched']
    
    test_matched_data = data['test_matched']
    test_mismatched_data = data['test_mismatched']
    
    if mode == "test" :
        if config.get("data", "matched"):
            data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_matched_data]
        elif not config.get("data", "matched"):
            data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_mismatched_data]
    elif mode == "valid":
        if config.get("data", "matched"):
            data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_matched_data]
        elif not config.get("data", "matched"):
            data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_mismatched_data]
    else:
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in train_data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
            
    return data, len(data)


#####################아직안고침#############################################

#"./data/laptop/train.json
# if local map : emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
# else : emo_dict={"positive":7,"netural":6,"negative":5,"conflict":8} 
def pre_data_laptop(mode,config):
    
    local_map = True
    
    if not local_map:
        emo_dict={"positive":7,"neutral":6,"negative":5,"conflict":8}
    else :
        emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
    
    dataset_size = config.getint("train","dataset_size")
    
    
    if mode == "train":
        # data = json.load(open("./data/laptop/train.json", "r"))
        data = []
        with open("./data/laptop/train.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif mode == "valid":
        # data = json.load(open("./data/laptop/test.json", "r"))
        data = []
        with open("./data/laptop/test.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        # data = json.load(open("./data/laptop/test.json", "r"))
        data = []
        with open("./data/laptop/test.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    if mode == "test":
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "dataset":"laptop"})
        # data = [{"sent1": ins['sentence'].strip()+ " " + ins["aspect"].strip(), "dataset":"laptop"} for ins in data]
    elif mode == 'valid':
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "label": emo_dict[ins['aspects']["polarity"][0]], "dataset":"laptop"})
        # data = [{"sent1": ins['sentence'].strip()+ " " + ins["aspect"].strip(), "label": emo_dict[ins['sentiment']], "dataset":"laptop"} for ins in data]
    else:
        self_data = []
        for ins in data:
            if ins["aspects"]["term"][0] != "":
                self_data.append({"sent1": ins['text'].strip()+ " " + ins["aspects"]["term"][0].strip(), "label": emo_dict[ins['aspects']["polarity"][0]], "dataset":"laptop"})
            # data = [{"sent1": ins['sentence'].strip()+ " " + ins["aspect"].strip(), "label": emo_dict[ins['sentiment']], "dataset":"laptop"} for ins in data]
        
        if dataset_size == -1 :
            pass
        else :
            self_data = self_data[:dataset_size]
        
            
    return self_data, len(self_data)



#("./data/laptop/train.json"
# if local_map : {0:0, 1;1} else {0:5,1:7}
def pre_data_imdb(mode,config,data_name=None):

    local_map = True
    if not local_map:
        label_map = {0:5,1:7}
    else : 
        label_map = {0:0,1:1}
        
    dataset_size = config.getint("train","dataset_size")
        
        
    if mode == "train":
        data_imdb = csv.reader(open("./data/IMDB/train.csv", "r"), delimiter='\t')
    elif mode == "valid":
        data_imdb = csv.reader(open("./data/IMDB/dev.csv", "r"), delimiter='\t')
    else:
        data_imdb = csv.reader(open("./data/IMDB/test.csv", "r"), delimiter='\t')

    data = [row for row in data_imdb]
    
    if mode == "test":
        data = [{"sent1": ins[0].strip(), "dataset":"imdb"} for ins in data]
    elif mode == "train":
        
        data = [{"sent1": ins[0][:-2].strip(), "label": label_map[int(ins[0][-1].strip())], "dataset":"imdb"} for ins in data[1:]]
        
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    else :
        data = [{"sent1": ins[0][:-2].strip(), "label": label_map[int(ins[0][-1].strip())], "dataset":"imdb"} for ins in data[1:]]

    return data, len(data)


#./data/snli/train.json
# if local_map : {0:2,1:1,2:0} else {2:0,1:3,0:1}
def pre_data_snli(mode,config):
    local_map = True
    if not local_map :
        _dict={2:0,1:3,0:1}
    else :
        _dict = {0:2,1:1,2:0}
    
    dataset_size = config.getint("train","dataset_size")
    
    data = []
    if mode == "train":
        with open("./data/snli/train.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif mode == "valid":
        with open("./data/snli/dev.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open("./data/snli/test.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    if mode == "test":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"snli"} for ins in data]
    
    elif mode == "train":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])], "dataset":"snli"} for ins in data if int(ins["label"])!=-1]
        
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    else:
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])], "dataset":"snli"} for ins in data if int(ins["label"])!=-1]
        
    return data, len(data)

#./data/anli/train.json -> but we do not have!
# if local_map : {0:2,1:1,2:0} else {2:0,1:3,0:1}
def pre_data_anli(mode,config):
    
    local_map = True
    if not local_map :
        _dict={2:0,1:3,0:1}
    else :
        _dict = {0:2,1:1,2:0}
    
    dataset_size = config.getint("train","dataset_size")
    
    if mode == "train":
        data = json.load(open("./data/anli/train.json"))
    elif mode == "valid":
        data = json.load(open("./data/anli/dev.json"))
    else:
        data = json.load(open("./data/anli/test.json"))
    
    if mode == "test":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"snli"} for ins in data]
    elif mode == "train":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])],"dataset":"snli"} for ins in data if int(ins["label"])!=-1]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    else:
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])],"dataset":"snli"} for ins in data if int(ins["label"])!=-1]
    # from IPython import embed; embed()
    return data, len(data)


#./data/tweeteval/sentiment/train.json
#if local_map : {"positive":2,"neutral":1,"negative":0,}
#else {"positive":7,"neutral":3,"negative":5}
def pre_data_tweetevalsentiment(mode,config):
    local_map = True
    if not local_map :
        emo_dict={"positive":7,"neutral":3,"negative":5}
    else :
        emo_dict={"positive":2,"neutral":1,"negative":0}
        
    dataset_size = config.getint("train","dataset_size")
    
    data = []
    if mode == "train":
        with open("./data/tweeteval/sentiment/train.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    elif mode == "valid":
        # data = json.load(open("./data/tweeteval/sentiment/dev.json"))
        with open("./data/tweeteval/sentiment/dev.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        # data = json.load(open("./data/tweeteval/sentiment/test.json"))
        with open("./data/tweeteval/sentiment/train.json", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    if mode == "test":
        data = [{"sent1": ins['text'].strip(), "dataset":"tweetevalsentiment"} for ins in data]
    elif mode == 'valid':
        data = [{"sent1": ins['text'].strip(), "label": int(ins['label']), "dataset":"tweetevalsentiment"} for ins in data]
    else:
        data = [{"sent1": ins['text'].strip(), "label": int(ins['label']), "dataset":"tweetevalsentiment"} for ins in data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    return data, len(data)

#./data/movie-rationales/train.json
# if local_map : {0:0, 1;1} else {0:5,1:7}
def pre_data_movierationales(mode,config):
    
    local_map = True
    if not local_map:
        emo_dict = {1:7, 0:5}
    else : 
        emo_dict= {0:0,1:1}
    
    dataset_size = config.getint("train","dataset_size")
    
    if mode == "train":
        data = json.load(open("./data/movie-rationales/train.json"))
    elif mode == "valid":
        data = json.load(open("./data/movie-rationales/dev.json"))
    else:
        data = json.load(open("./data/movie-rationales/test.json"))
    if mode == "test":
        data = [{"sent1": ins['review'].strip(), "dataset":"movierationales"} for ins in data]
    elif mode == 'valid':
        data = [{"sent1": ins['review'].strip(), "label": emo_dict[int(ins['label'])], "dataset":"movierationales"} for ins in data]
    else:
        data = [{"sent1": ins['review'].strip(), "label": emo_dict[int(ins['label'])], "dataset":"movierationales"} for ins in data]
        
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    return data, len(data)


#./data/ethics/justice/justice_train.csv -> but we do not have!
# if local_map : {0:0,1:1} else {0:9,1:10]}
def pre_data_ethicsjustice(mode,config):
    
    local_map = True
    if not local_map:
        label_map = {0:9, 1:10}
    else : 
        label_map = {0:0,1:1}
    
    dataset_size = config.getint("train","dataset_size")
    
    if mode == "train":
        data = csv.reader(open("./data/ethics/justice/justice_train.csv"), delimiter=",")
    elif mode == "valid":
        data = csv.reader(open("./data/ethics/justice/justice_test.csv"), delimiter=",")
    else:
        data = csv.reader(open("./data/ethics/justice/justice_test.csv"), delimiter=",")
    
    data = [row for row in data if row[0]=='1' or row[0]=='0']
    
    if mode == "test":
        data = [{"sent1": ins[1].strip(),"dataset":"ethicsjustice"} for ins in data]
    
    elif mode == "train":
        data = [{"sent1": ins[1].strip(), "label": label_map[int(ins[0])], "dataset":"ethicsjustice"} for ins in data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    
    else:
        data = [{"sent1": ins[1].strip(), "label": label_map[int(ins[0])], "dataset":"ethicsjustice"} for ins in data]

    return data, len(data)


#./data/ethics/deontology/deontology_train.csv -> but we do not have!
# if local_map : {0:0,1:1} else {0:9,1:10]}
def pre_data_ethicsdeontology(mode,config):
    
    local_map = True
    if not local_map:
        label_map = {0:9, 1:10}
    else : 
        label_map = {0:0,1:1}
    
    dataset_size = config.getint("train","dataset_size")
    
    if mode == "train":
        data = csv.reader(open("./data/ethics/deontology/deontology_train.csv"), delimiter=",")
    elif mode == "valid":
        data = csv.reader(open("./data/ethics/deontology/deontology_test.csv"), delimiter=",")
    else:
        data = csv.reader(open("./data/ethics/deontology/deontology_test.csv"), delimiter=",")
    #_map = {"low":9, "high":10}
    data = [row for row in data if row[0]=='1' or row[0]=='0']
    #data = [row for row in fin]
    if mode == "test":
        data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(),"dataset":"ethicsdeontology"} for ins in data]
    
    elif mode == "train":
        data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": int(ins[0]),"dataset":"ethicsdeontology"} for ins in data]
        if dataset_size == -1 :
            pass
        else :
            data = data[:dataset_size]
        
    else:
        data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": int(ins[0]),"dataset":"ethicsdeontology"} for ins in data]
    return data, len(data)


