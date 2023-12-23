import torch
import os

name_list = ["snli","sst2","qqp"]
name_list.sort()
model = "Bert"
output_name = ""


projector_list = []

for name in name_list:
    path = "task_prompt_emb/"+name+"Prompt"+model+"/task_prompt"
    projector_list.append(torch.load(path))
    output_name = output_name + name+"_"
    
result = torch.zeros_like(projector_list[0])
for tensor in projector_list:
    result += tensor


result = result / torch.tensor(len(name_list))

output_path = "task_prompt_emb/"+output_name+"Prompt"+model
print("mixing prompt ... {}".format(name_list))
try:
    os.makedirs(output_path)
    torch.save(result,output_path+"/task_prompt")
    print("success mixing prompt {} to <{}>".format(name_list,output_path))
except:
    print("already exist mix prompt of {}".format(name_list))



