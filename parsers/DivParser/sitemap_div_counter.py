from bs4 import BeautifulSoup
from langchain_community.document_loaders.sitemap import SitemapLoader

SITEMAP = "https://www.towson.edu/sitemap.xml"

loader = SitemapLoader(SITEMAP, continue_on_failure=True)
documents = loader.load()

div_info = []

for document in documents:
    soup = BeautifulSoup(document.page_content, 'html.parser')
    divs = soup.find_all('div')
    for div in divs:
        div_class = div.get('class', [])
        div_id = div.get('id', [])
        div_info.append({
            'class': div_class,
            'id': div_id
        })

# Save div_info to a text file
with open('div_info.txt', 'w', encoding='utf-8') as file:
    for div in div_info:
        class_names = ', '.join(div['class'])
        div_id = div['id'][0] if div['id'] else ''
        file.write(f"Class: {class_names}, ID: {div_id}\n")



