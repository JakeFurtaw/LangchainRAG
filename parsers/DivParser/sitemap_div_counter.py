from bs4 import BeautifulSoup
from langchain_community.document_loaders.sitemap import SitemapLoader

SITEMAP = "https://www.towson.edu/sitemap.xml"

loader = SitemapLoader(SITEMAP, continue_on_failure=True)
documents = loader.load()

div_classes = []
div_ids = []

for document in documents:
    content= document.page_content
    print(content[:100])
    soup = BeautifulSoup(content, 'html.parser')
    divs = soup.find_all('div')
    for div in divs:
        if div.get('class'):
            div_classes.extend(div.get('class'))
        if div.get('id'):
            div_ids.append(div.get('id'))

print(f"Number of div classes: {len(div_classes)}")
print(f"Number of div ids: {len(div_ids)}")
print("Div classes:", div_classes)
print("Div ids:", div_ids)




