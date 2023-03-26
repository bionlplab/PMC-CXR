import numpy as np
import json
import pandas as pd
pmcid = []
caption = []
with open('/prj0129/mil4012/glaucoma/PMCFigureX/Pneumonia/Pneumonia.figure_text.json','r') as file:
    str = file.read()
    data = json.loads(str)
    for i in range(len(data)):
        caption = np.append(caption,data[i]['caption']['text'])
        pmcid = np.append(pmcid,(data[i]['pmcid'][:-4] + '/' + data[i]['pmcid'][-4:-2] +'/' + data[i]['pmcid'] + '_' + data[i]['caption']['infons']['file']))
   
    caption  = np.reshape(caption,(len(caption),1))
    pmcid = np.reshape(pmcid,(len(pmcid),1))
    list=np.concatenate((pmcid,caption),axis=1)
    column=['ID','TEXT']
    lab=pd.DataFrame(columns=column,data=list)
    lab.to_csv("/prj0129/mil4012/glaucoma/PMCFigureX/Pneumonia/Pneumonia.csv",index=False)