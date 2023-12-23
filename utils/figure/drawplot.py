import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import os
import json

#https://github.com/mousepixels/sanbomics_scripts/tree/main
#https://github.com/mousepixels/sanbomics_scripts/blob/main/high_quality_lineplots.ipynb

os.makedirs("figure",exist_ok=True)
json_name = "mnli_sst2_qqp_RobertaBase"
PATH = "result/" + json_name

length = 50

with open(PATH,'r',encoding='utf-8') as file:
    train_valid_info = json.load(file)



dataList = []
for key in train_valid_info:
    if not length :
        length = len(train_valid_info[key])
    if 'train_loss' in key:
        dataList.append(key)

data_num = len(dataList)

epoch = []
for x in range(1,length+1): 
    epoch += [x]

for _ in range(data_num):
    epoch += epoch

vals = []
labels = []
for dataname in dataList:
    datadict = train_valid_info[dataname]
    for j in range(1,length+1):
        vals = vals + [datadict[str(j)]]
        labels = labels + [dataname.replace('_train_loss','')]

df = pd.DataFrame(zip(vals, epoch, labels), columns = ['Loss', 'Epoch', 'data'])
df.head()

plt.figure(figsize = (4,4))

#err_kws = {'capsize': 5, 'capthick': 2, 'elinewidth':2}


ax = sns.lineplot(data = df, x = 'Epoch', y = 'Loss', hue = 'data', lw = 2.5,
                 style = 'data', dashes = False, markersize = 8 ,
                  palette = ['green','gray', 'firebrick'])

#, markers = ['o','o','o']
# err_style = 'bars', err_kws = err_kws,
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width = 2.5, color = '0.2')

plt.xticks(size = 14, weight = 'bold', color = '0.2')
plt.yticks(size = 14, weight = 'bold', color = '0.2')

ax.set_xlabel(ax.get_xlabel(), fontsize = 14, weight = 'bold', color = '0.2')
ax.set_ylabel(ax.get_ylabel(), fontsize = 14, weight = 'bold', color = '0.2')

plt.legend(frameon = False, prop = {'weight':'bold', 'size':10}, labelcolor = '0.2')

plt.savefig('figure/'+json_name+".png", bbox_inches = 'tight', dpi = 250, facecolor = ax.get_facecolor())