import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import os
import json

#https://github.com/mousepixels/sanbomics_scripts/tree/main
#https://github.com/mousepixels/sanbomics_scripts/blob/main/high_quality_lineplots.ipynb

def drawplot_valid_acc(length,json_name,nbins):
    os.makedirs(json_name,exist_ok=True)

    with open("./../result/"+json_name,'r',encoding='utf-8') as file:
        train_valid_info = json.load(file)

    dataList = []
    for key in train_valid_info:
        if not length :
            length = len(train_valid_info[key])
        if 'valid_acc' in key:
            dataList.append(key)

    data_num = len(dataList)
    if data_num == 0:
        return None
    
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
            labels = labels + [dataname.replace('_valid_acc','')]

    df = pd.DataFrame(zip(vals, epoch, labels), columns = ['Accuracy', 'Epoch', 'data'])
    df.head()

    # plt.figure(figsize = (16,16))
    plt.figure(figsize = (4,4))


    #err_kws = {'capsize': 5, 'capthick': 2, 'elinewidth':2}


    ax = sns.lineplot(data = df, x = 'Epoch', y = 'Accuracy', hue = 'data', lw = 2.5,
                    style = 'data', dashes = False, markersize = 8 ,
                    )
    #palette = ['green','gray', 'firebrick']
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

    # x_ticks_interval = 5
    plt.locator_params(axis='x', nbins=nbins )
    plt.locator_params(axis='y', nbins=6 )

    ax.set_xlabel(ax.get_xlabel(), fontsize = 14, weight = 'bold', color = '0.2')
    ax.set_ylabel(ax.get_ylabel(), fontsize = 14, weight = 'bold', color = '0.2')

    plt.grid(True,linestyle = '--',linewidth=0.5)
    plt.legend(frameon = True, prop = {'weight':'bold', 'size':5}, labelcolor = '0.2',loc='upper left')

    title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
    }
    plt.title('Valid Accuracy', fontdict=title_font, loc='left', pad=20)

    plt.savefig(json_name+"/valid_acc.png", bbox_inches = 'tight', dpi = 250, facecolor = ax.get_facecolor())