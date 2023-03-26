# PMC-CXR

## Download code

Go to this website to download the related code to get figure from PubMed https://github.com/yfpeng/PMCFigureX

Go to this website to download the code for subfigure separation https://github.com/hrlblab/ImageSeperation




## Prepare source file

1. Go to https://pubmed.ncbi.nlm.nih.gov/
2. Search disease. For example Atelectasis [all_field]. Note: PubMed will automatically find synonyms of atelectasis, e.g., "pulmonary atelectasis"       [MeSHTerms] OR ("pulmonary"[All Fields] AND "atelectasis"[All Fields]) OR "pulmonary  atelectasis"[All Fields] OR "atelectasis"[All Fields]
3. On the left, click "Free full text"
4. Click "Save" and choose the "CSV" format: /path/to/Atelectasis.export.csv

## Convert PubMed export file

$ python figurex_db/convert_pubmed_search_output.py \
    -s /path/to/Atelectasi.export.csv \
    -d /path/to/Atelectasi.export.tsv
    
    
## Run the script

Change the paths in run_keys_db.sh

disease='Atelectasis'
source_dir=$HOME'/path/to/PMCFigureX'
venv_dir=$HOME'/path/to/venv'
top_dir=$HOME'/path/to/Atelectasi.export.tsv'

##run the bash file: Create database, Get PMC ID from PubMed, Get BioC files, Get figures, and Download local figures

bash run_keys_db.sh step1 step2 step3 step4 step5 

##To generate COCO dataset format for image segmentation. 

python generate_coordinate.py  

##To get image size
python generate_coordinate.py 

##Subfigure sepearation
python detect.py --weights /prj0129/mil4012/glaucoma/Figure_segmentation/runs/train/exp6/weights/best.pt --source /prj0129/mil4012/glaucoma/Figure_segmentation/Pneumonia/images/test --hide-labels --hide-conf --save-txt --save-conf

##Get the json file

python save_jsonnew.py 

##Get Get local figures/subfigures, Classify subfigures, and Get text

bash run_keys_db.sh step7 step8 step9

##Get the second classification result for figure 

python classifier_second.py

##Produce a CSV file for Radtex, which will be used to verify CXR pathology

python create_csv.py

##CXR pathology verification

Please follow the Process.doc to generate result for CXR pathology verification.

We will maintain and update the RadText software (https://github.com/bionlplab/radtext).

## Get the PMC data

In this study, we created the PMC-CXR based on the following three criteria: (1) the caption contains a positive mention of the disease (CXR pathology verification), (2) the figure/subfigure is a chest x-ray (CXR) (two classifiers identify the image is  a chest x-ray), and (3) the subfigure has a width-to-height or height-to-width ratio greater than 0.5.






