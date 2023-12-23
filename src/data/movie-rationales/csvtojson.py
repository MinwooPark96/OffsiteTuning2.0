import csv
import json

# CSV 파일 경로
csv_file_path = 'validation.csv'
# JSON 파일 경로
json_file_path = 'dev.json'

# CSV 파일을 읽어 JSON 파일로 저장
data = []

with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# JSON 파일로 데이터 저장
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4)