import json
from pathlib import Path

txt_path = Path("data/book.txt")
jsonl_path = Path("data/arxiv.jsonl") 
if not txt_path.exists():
    print(f"Файл {txt_path} не знайдено! Поклади book.txt у папку data/")
    exit(1)

print("Конвертація тексту у формат jsonl...")
with open(txt_path, "r", encoding="utf-8", errors="ignore") as f_in, \
     open(jsonl_path, "w", encoding="utf-8") as f_out:
    
    doc_id = 1
    for line in f_in:
        text = line.strip()
        if len(text) > 20: 
            record = {"id": f"doc_{doc_id}", "abstract": text}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_id += 1

print(f"Готово! Створено {doc_id - 1} документів у файлі data/arxiv.jsonl")