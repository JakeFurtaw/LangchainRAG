from bs4 import BeautifulSoup
from collections import Counter
from langchain_community.document_loaders.sitemap import SitemapLoader

def find_repeating_div_names(sitemap_url, output_file):
    loader = SitemapLoader(sitemap_url, continue_on_failure=True)
    docs = loader.load()

    all_div_attributes = []

    for doc in docs:
        html_content = doc.page_content
        soup = BeautifulSoup(html_content, "html.parser")
        divs = soup.find_all("div")
        
        div_attributes = []
        for div in divs:
            class_names = div.get('class', [])
            id_name = div.get('id', '')
            div_attributes.append({
                'class': class_names,
                'id': id_name
            })
        
        all_div_attributes.extend(div_attributes)

    print("Number of div elements found:", len(all_div_attributes))
    print("Number of unique div classes found:", len(set(attr['class'] for attr in all_div_attributes)))
    print("Number of unique div IDs found:", len(set(attr['id'] for attr in all_div_attributes if attr['id'])))

    class_counter = Counter(cls for attr in all_div_attributes for cls in attr['class'])
    id_counter = Counter(attr['id'] for attr in all_div_attributes if attr['id'])

    with open(output_file, "w") as file:
        file.write("Repeating class names:\n")
        for class_name, count in class_counter.most_common():
            if count > 5:
                file.write(f"{class_name}: {count}\n")
        
        file.write("\nRepeating IDs:\n")
        for div_id, count in id_counter.most_common():
            if count > 5:
                file.write(f"{div_id}: {count}\n")

sitemap_url = "https://www.towson.edu/sitemap.xml"
output_file = "div_stats.txt"
find_repeating_div_names(sitemap_url, output_file)