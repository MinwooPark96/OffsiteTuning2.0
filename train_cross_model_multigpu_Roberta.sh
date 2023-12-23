


gpus="1,2,3,4,5"

# # PROMPT_EMB="sst2PromptT5"

TARGET=Roberta

for SOURCE in Bert
    do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$gpus torchrun --nnodes 1 --nproc-per-node 5 train_cross.py \
        --config config/develop_for_MTL_${SOURCE}_${TARGET}.config \
        --gpu $gpus \
        --source_model ${SOURCE}\
        --seed 42\
        # --prompt_emb ${PROMPT_EMB}\
    done
    
