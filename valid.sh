
# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large

mkdir PLM
mkdir PLM/BertMediumForMaskedLM
mkdir PLM/BertForMaskedLM
mkdir PLM/BertLargeForMaskedLM

mkdir PLM/RobertaForMaskedLM
mkdir PLM/RobertaLargeForMaskedLM

mkdir PLM/T5SmallForMaskedLM
mkdir PLM/T5ForMaskedLM
mkdir PLM/T5LargeForMaskedLM
mkdir PLM/T53BForMaskedLM
mkdir PLM/T511BForMaskedLM


gpus="4"

SOURCE = "Bert"

CUDA_VISIBLE_DEVICES=$gpus python3 valid.py \
    --config config/develop_for_valid.config \
    --gpu $gpus \
    --prompt_emb mrpcPrompt${SOURCE}\
    --seed 28 \
    --projector model/projector_hub/15_model_cross_0.8896.pkl\
    
