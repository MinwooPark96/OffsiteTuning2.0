from datasets import load_dataset
import json

# 데이터셋을 로드합니다. 여기에서는 IMDb 데이터셋을 예로 들었습니다.
dataset = load_dataset("jakartaresearch/semeval-absa", 'restaurant',split="train")
test_dataset = load_dataset("jakartaresearch/semeval-absa", 'restaurant',split="validation")

# 데이터를 JSON 파일로 저장합니다.
dataset.to_json("train.json")
test_dataset.to_json("test.json")