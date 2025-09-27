from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
import requests
from bs4 import BeautifulSoup

def scrape_bbc_headlines():
    url = "https://www.bbc.com/news"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    documents = []
    for heading in soup.select("h3"):
        content = heading.get_text(strip=True)
        documents.append({
            'page_content': content,
            'metadata': {'source': 'BBC'}
        })
    return documents

def process_all_docs(data_directory):
    all_documents = []
    try:
        # Load local TXT documents
        txt_loader = DirectoryLoader(data_directory, glob="**/*.txt", loader_cls=TextLoader, show_progress=False)
        all_documents.extend(txt_loader.load())
        # Load local PDF documents
        pdf_loader = DirectoryLoader(data_directory, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=False)
        all_documents.extend(pdf_loader.load())
        # Add scraped BBC headlines
        news_docs = scrape_bbc_headlines()
        all_documents.extend(news_docs)
        print(f"✓ Loaded All {len(all_documents)} documents")
    except Exception as e:
        print(f"✗ Error: {e}")
    return all_documents

def search_documents(all_documents, query, max_results=10):
    results = []
    query_lower = query.lower()
    for doc in all_documents:
        # Support both dict-type docs (scraped) and LangChain Documents (files)
        content = doc['page_content'] if isinstance(doc, dict) else doc.page_content
        if query_lower in content.lower():
            results.append(doc)
        if len(results) >= max_results:
            break
    return results
