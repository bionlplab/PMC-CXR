import os
import cv2
import numpy as np
import json

detect_path = '/prj0129/mil4012/glaucoma/Figure_segmentation/runs/detect/exp17/labels'
save_path = '/prj0129/mil4012/glaucoma/Figure_segmentation/lung lesion'
jason_path = '/prj0129/mil4012/glaucoma/PMCFigureX/bioc_lung lesion'
detect_list = os.listdir(detect_path)

total = 0
for i in range(len(detect_list)):
    print(detect_list[i])
    index1 = detect_list[i].index('C')
    index2 = detect_list[i].index('_')
    pmc = detect_list[i][index1+1:index2]
#     print(detect_list[i])
    # the path to save json file
    json_path1 = os.path.join(jason_path,(detect_list[i][0:index1+1] + pmc[:-4] + '/' + pmc[-4:-2] + '/' + detect_list[i][:-4] + '.json'))
#     print('the json_path is', json_path1)
    
    # load the image size: height, width, channel
    image_size = np.loadtxt(os.path.join(save_path,'labels/test',(detect_list[i][:-4]+'.txt')))
#     print('the image size is', image_size)
    
    # load the file to get the detection result.
    detect_inf = np.loadtxt(os.path.join(detect_path,detect_list[i]),ndmin=2)
#     print('the detect_inf is', detect_inf)
    
    
    outboxes = []
    for j in range(len(detect_inf)):
        detect_in = detect_inf[j]
#       print('the detect_in is', detect_in)
        a = detect_in[1]
        b = detect_in[2]
        c = detect_in[3]
        d = detect_in[4]
        prob = detect_in[5]
        xtl = int(round((2*a - c) * image_size[1]))
        ytl = int(round((2*b - d) * image_size[0])) 
        w = int(round(c * image_size[1]))
        h = int(round(d * image_size[0]))
        outboxes.append({"x": xtl, "y": ytl, "w": w, "h": h, "conf": float(prob)})
    json.dump(outboxes, open(json_path1, 'w'))
    total += 1
    
print('the total is', total)    