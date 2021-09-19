# Evidence-based Factual Error Correction with Fusion in Decoder

The code is mainly based on https://github.com/j6mes/acl2021-factual-error-correction and https://github.com/facebookresearch/FiD

## Requirements

```bash
pip install pytorch-lightning==0.8.5
pip install transformers==3.5.1
pip install torch==1.7
pip install rouge-score
pip install lime
pip install fever-drqa
pip install tqdm
pip install pymysql
pip install seqeval
pip install numpy
pip install torchtext==0.8
```

## Data
The data are located in this Google Drive folder https://drive.google.com/open?id=1hzwg5NtVUB_cfXiADkSanCq0JjaQ87tV

## Supervised (retriever + corrector)

### Training
```bash
bash ../scripts/finetune_supervised_pipeline.sh t5-small 1e-4 true false all genre_50_2 train
```
### Testing
```bash
bash ../scripts/finetune_supervised_pipeline.sh t5-small 1e-4 true false all genre_50_2 test
```

## Distant (retriever + masker + corrector)
### Training
```bash
bash ../scripts/finetune_masked_pipeline.sh t5-small 1e-4 random_masked_50 all genre_50_2 train
```
### Testing
```bash
bash ../scripts/finetune_masked_pipeline.sh t5-small 1e-4 random_masked_50 all genre_50_2 test
```

### Notes
Scripts have to be run while being in the src directory
Data paths have to be altered from the scripts