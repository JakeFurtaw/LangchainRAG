from bs4 import BeautifulSoup
from langchain_community.document_loaders.sitemap import SitemapLoader
from collections import Counter

SITEMAP = "https://www.towson.edu/sitemap.xml"

loader = SitemapLoader(SITEMAP, continue_on_failure=True)
documents = loader.load()

div_classes = []
div_ids = []

for document in documents:
    content = document.page_content
    soup = BeautifulSoup(content, 'html.parser')
    divs = soup.find_all('div')
    for div in divs:
        if div.get('class'):
            div_classes.extend(div['class'])
        if div.get('id'):
            div_ids.append(div['id'])

class_counts = Counter(div_classes)
id_counts = Counter(div_ids)
print(f"Number of unique div classes: {len(class_counts)}")
print("Div class counts:")
for class_name, count in class_counts.items():
    if count > 100:
        print(f"'{class_name}': {count}")

print(f"\nNumber of unique div ids: {len(id_counts)}")
print("Div id counts:")
for div_id, count in id_counts.items():
    if count > 100:
        print(f"'{div_id}': {count}")




