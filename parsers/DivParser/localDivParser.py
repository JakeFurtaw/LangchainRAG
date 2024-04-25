from bs4 import BeautifulSoup
from pathlib import Path
from collections import Counter

div_classes = []
div_ids = []
directory = Path('/home/jake/Programming/LangchainRAG-Chatbot/parsers/HTMLtoText')


for file_path in directory.glob('*.html'):
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    divs = soup.find_all('div')
    for div in divs:
        if div.get('class'):
            div_classes.extend(div.get('class'))
        if div.get('id'):
            div_ids.append(div.get('id'))


class_counts = Counter(div_classes)
id_counts = Counter(div_ids)
print(f"Number of unique div classes: {len(class_counts)}")
print("Div class counts:")
for class_name, count in class_counts.items():
    if count > 10:
        print(f"'{class_name}': {count}")

print(f"\nNumber of unique div ids: {len(id_counts)}")
print("Div id counts:")
for div_id, count in id_counts.items():
    if count > 10:
        print(f"'{div_id}': {count}")