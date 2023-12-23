
# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large

gpus="0"

SOURCE=Bert
TARGET=Roberta
OPTION=cs_01_1e4_dev
DATAS=imdb

mkdir valid_result
mkdir valid_result/${DATAS}_${SOURCE}_${TARGET}_${OPTION}

for (( EPOCH=1; EPOCH<=20; EPOCH+=1))
do
    for DATASET in imdb sst2 laptop restaurant movierationales tweetevalsentiment #mnli qnli snli ethicsdeontology ethicsjustice qqp mrpc
    do
    
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py \
        --config config/valid_configs_${SOURCE}_${TARGET}_auto/${DATASET}.config \
        --gpu $gpus \
        --prompt_emb ${DATASET}Prompt${SOURCE}  \
        --projector checkpoint/${DATAS}_${SOURCE}_${TARGET}_${OPTION}/${EPOCH}_model_cross.pkl\
        --output_name valid_result/${DATAS}_${SOURCE}_${TARGET}_${OPTION}/${EPOCH}\
        --seed 42
    done 
done
