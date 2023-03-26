# PMC-CXR

## Download code

Go to this website to download the related code https://github.com/yfpeng/PMCFigureX


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

##


