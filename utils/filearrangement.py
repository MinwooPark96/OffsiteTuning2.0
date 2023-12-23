import os

# PATH = "model/crossPromptTraining/"
PATH = "model/sst2_mnli_qqp_each_lr0.005/"

file_list = os.listdir(PATH)

file_list.sort()

for file in file_list:
    epoch = int(file.split('_')[0])
    if epoch % 10 != 0 :
        os.remove(PATH + file)