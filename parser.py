from newsplease import NewsPlease
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

def find_latest_articles(base_url, max_articles=5, max_age_days=7):
    print(f"[PARSER] Crawling {base_url} for latest articles...")
    
    try:
        response = requests.get(base_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if urlparse(full_url).netloc == base_domain:
                if not any(x in full_url.lower() for x in ['#', 'javascript:', 'mailto:', '/tag/', '/category/', '/author/']):
                    links.append(full_url)
        
        links = list(dict.fromkeys(links))[:max_articles * 3] 
        
        print(f"[PARSER] Found {len(links)} potential article links")
        return links[:max_articles]
        
    except Exception as e:
        print(f"[PARSER] ERROR crawling {base_url}: {e}")
        return []

def extract_from_sites(site_urls, max_articles_per_site=5, replacements=None):
    all_article_urls = []
    
    for site_url in site_urls:
        article_urls = find_latest_articles(site_url, max_articles=max_articles_per_site)
        all_article_urls.extend(article_urls)
    
    print(f"[PARSER] Total articles to extract: {len(all_article_urls)}")
    
    return extract(all_article_urls, replacements)

def extract(urls, replacements):
    print(f"[PARSER] Extracting {len(urls)} articles...")
    articles = NewsPlease.from_urls(urls, request_args={"timeout": 10})
    content = ""
    count = 0
    
    for url, article in articles.items():
        count += 1
        if article and article.maintext:
            content += f"[FRAMEWORK] ARTICLE {count} BEGINN\n"
            content += f"Title: {article.title}\n"
            content += f"Authors: {', '.join(article.authors) if article.authors else 'N/A'}\n"
            content += f"Date: {article.date_publish}\n"
            content += f"Description: {article.description}\n"
            content += f"Text:\n{article.maintext}\n"
            content += f"[FRAMEWORK] ARTICLE {count} END\n\n"
            print(f"[PARSER] ✓ Extracted: {article.title}")
        else:
            content += f"[FRAMEWORK] ERROR: Failed to extract article {count} from {url}\n\n"    
            print(f"[PARSER] ✗ Failed to extract article from {url}")

    if replacements:
        for replacement in replacements:
            content = content.replace(replacement, "")

    print(f"[PARSER] Extraction complete: {count} articles processed")
    return content