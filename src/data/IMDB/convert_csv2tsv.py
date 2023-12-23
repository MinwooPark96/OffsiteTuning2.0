import csv

# CSV 파일 경로
csv_file_path = 'train.csv'
# TSV 파일 경로
tsv_file_path = 'train.tsv'

# CSV 파일을 읽어 TSV 파일로 저장
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open(tsv_file_path, 'w', newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            tsv_writer.writerow(row)