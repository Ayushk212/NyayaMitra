import json
import urllib.parse
import concurrent.futures
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
dataset_path = str(Path(__file__).resolve().parent.parent / "dataset.json")
seed_cases_path = str(Path(__file__).resolve().parent.parent / "data" / "seed_cases.json")

def scrape_clean_url(url):
    url = url.split("?")[0]
    url = url.replace("/docfragment/", "/doc/")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            judgments_div = soup.find('div', class_='judgments')
            if judgments_div:
                text = judgments_div.get_text(separator='\n\n', strip=True)
                return text[:4000]
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    return None

def fix_dataset():
    print("Fixing dataset.json...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    needs_fix = []
    for c in dataset:
        if "Skip to main content" in c.get("full_text", "") and c.get("source_url"):
            needs_fix.append(c)
            
    print(f"Found {len(needs_fix)} cases in dataset.json needing re-scrape.")
    
    def process_case(c):
        text = scrape_clean_url(c["source_url"])
        if text and "Skip to main content" not in text:
            c["full_text"] = text
            c["summary"] = text[:350].replace('\n', ' ').strip() + "..."
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(process_case, needs_fix))
        
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print("dataset.json fixed!")

def get_url_from_search(title):
    try:
        url = f"https://indiankanoon.org/search/?formInput={urllib.parse.quote(title)}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.text, 'html.parser')
        res = soup.find('div', class_='result_title')
        if res and res.a:
            return "https://indiankanoon.org" + res.a['href'].replace('/docfragment/', '/doc/').split('?')[0]
    except:
        pass
    return None

def fix_seed_cases():
    print("Fixing seed_cases.json...")
    with open(seed_cases_path, "r", encoding="utf-8") as f:
        seed_cases = json.load(f)
        
    for c in seed_cases:
        if len(c.get("full_text", "")) < 2000:
            print(f"Searching for {c['title']}...")
            url = get_url_from_search(c["title"])
            if url:
                text = scrape_clean_url(url)
                if text:
                    c["full_text"] = text
                    print(f"  -> Found and updated! Length: {len(text)}")
                else:
                    print(f"  -> Failed to extract text from {url}")
            else:
                print(f"  -> Could not find URL on IndianKanoon")
                
    with open(seed_cases_path, "w", encoding="utf-8") as f:
        json.dump(seed_cases, f, indent=4)
    print("seed_cases.json fixed!")

if __name__ == "__main__":
    fix_seed_cases()
    fix_dataset()
