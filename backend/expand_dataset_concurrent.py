import json
import re
import os
import concurrent.futures
from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-de9ac1a674d44cb2a8463b02ab2f9459")
output_file = r"c:\Users\ayush\Downloads\legal_dataset_expanded.json"

with open(output_file, "r", encoding="utf-8") as f:
    existing_data = json.load(f)

current_len = len(existing_data)
TARGET_TOTAL = 2000
TARGET_NEW = TARGET_TOTAL - current_len

if TARGET_NEW <= 0:
    print(f"Already reached {current_len}. Exiting...")
    exit(0)

seen_ids = set([c.get('case_id') for c in existing_data if 'case_id' in c])
found_ids = []

topics = ["murder", "property", "tax", "family", "contract", "corporate", "constitutional", "criminal", "civil", "fundamental+rights"]
topic_idx = 0
page_num = 0

print(f"Need {TARGET_NEW} new records. Fetching search indices heavily...")

while len(found_ids) < TARGET_NEW + 50 and topic_idx < len(topics):
    topic = topics[topic_idx]
    url = f"https://indiankanoon.org/search/?formInput={topic}+doctypes:supremecourt&pagenum={page_num}"
    try:
        search_doc = app.scrape(url)
        if not isinstance(search_doc, dict):
            search_doc = search_doc.model_dump() if hasattr(search_doc, 'model_dump') else vars(search_doc)
            
        md = search_doc.get('markdown', '')
        ids = list(set(re.findall(r'/(?:doc|docfragment)/(\d+)', md)))
        
        if not ids:
            topic_idx += 1
            page_num = 0
            continue
            
        for i in ids:
            if i not in seen_ids and i not in found_ids:
                found_ids.append(i)
        print(f"Topic '{topic}', Page {page_num}: Unique queue now {len(found_ids)}")
    except Exception as e:
        topic_idx += 1
        page_num = 0
        continue
        
    page_num += 1
    if page_num > 40: # Max 40 pages per topic just to heavily balance diversity
        topic_idx += 1
        page_num = 0

def scrape_case(doc_id):
    case_url = f"https://indiankanoon.org/doc/{doc_id}/"
    try:
        case_doc = app.scrape(case_url)
        if not isinstance(case_doc, dict):
            case_doc = case_doc.model_dump() if hasattr(case_doc, 'model_dump') else vars(case_doc)
        
        case_md = case_doc.get('markdown', '')
        case_meta = case_doc.get('metadata', {})
        title = case_meta.get('title', f"Case {doc_id}").replace(' - Indian Kanoon', '').strip()
        
        return {
            "case_id": doc_id,
            "title": title,
            "summary": case_md[:350].replace('\n', ' ').strip() + "...",
            "full_text": case_md[:4000],
            "source_url": case_url
        }
    except Exception as e:
        return None

new_cases = []
added = 0
print(f"Scraping {len(found_ids)} candidates concurrently (Max 6 workers) to strict hit {TARGET_NEW}...")

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(scrape_case, doc_id): doc_id for doc_id in found_ids}
    for future in concurrent.futures.as_completed(futures):
        if added >= TARGET_NEW:
            break
        res = future.result()
        if res:
            new_cases.append(res)
            added += 1
            if added % 50 == 0:
                print(f" + {added}/{TARGET_NEW}: {res['title'][:30]}...")
            
            # Periodically backup so we don't lose data on interruption
            if added % 200 == 0:
                print(f"Intermediate save of {added} records.")
                temp_data = existing_data + new_cases
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(temp_data, f, indent=4)

existing_data.extend(new_cases[:TARGET_NEW])

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, indent=4)

print(f"\nSUCCESS: Appended {len(new_cases[:TARGET_NEW])} records. Final Total: {len(existing_data)}")
