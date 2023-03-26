import os
import cv2
import numpy as np
import json
#save image
# save_path = '/prj0129/mil4012/glaucoma/Figure_segmentation/lung lesion/images/test'
#save image size
save_path = '/prj0129/mil4012/glaucoma/Figure_segmentation/lung lesion'
# data_path = '/prj0129/mil4012/glaucoma/Figure_segmentation/bioc'
data_path = '/prj0129/mil4012/glaucoma/PMCFigureX/bioc_lung lesion'
files = os.listdir(data_path)

total  = 0

# image_size = []
for files_name in files:
        
    image_path = os.path.join(data_path,files_name)
    
    images = os.listdir(image_path)

    for image_name in images:
#         rout_path = os.path.join(image_path,images[0])
        rout_path = os.path.join(image_path,image_name)
        fig = os.listdir(rout_path)   
        for i in range(len(fig)):
            if fig[i][-3:] == 'jpg':
#                 print(os.path.join(rout_path,fig[i]))
                im = cv2.imread(os.path.join(rout_path,fig[i]))
#                 print(os.path.join(rout_path,fig[i]))
                if im is not None:
                    image_size = np.shape(im)
                    # cv2.imwrite(os.path.join(save_path, fig[i]),im)
                    np.savetxt(os.path.join(save_path,'labels/test',(fig[i][:-4]+'.txt')),image_size)

                    total += 1

print('the total is', total)