import json
import re
import os
from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-de9ac1a674d44cb2a8463b02ab2f9459")

source_file = r"c:\Users\ayush\Downloads\legal_dataset_1000 (1).json"
output_file = r"c:\Users\ayush\Downloads\legal_dataset_expanded.json"

print(f"Loading existing dataset from {source_file}...")
with open(source_file, "r", encoding="utf-8") as f:
    existing_data = json.load(f)

print(f"Loaded {len(existing_data)} records. Searching dataset for Corporate Law cases...")

search_url = "https://indiankanoon.org/search/?formInput=technology+corporate+law+doctypes:supremecourt"
try:
    search_doc = app.scrape(search_url)
    markdown = search_doc.markdown if hasattr(search_doc, 'markdown') else search_doc.get('markdown', '')
except Exception as e:
    print(f"Failed to load search URL: {e}")
    markdown = ""

new_cases = []
# Just grab any doc IDs we find in the markdown!
ids = list(set(re.findall(r'/(?:doc|docfragment)/(\d+)', markdown)))

seen_ids = set([c.get('case_id') for c in existing_data if 'case_id' in c])
added = 0
TARGET_COUNT = 5

already_seen_in_loop = set()

for doc_id in ids:
    if added >= TARGET_COUNT:
        break
    if doc_id in seen_ids or doc_id in already_seen_in_loop:
        continue
    
    already_seen_in_loop.add(doc_id)
    case_url = f"https://indiankanoon.org/doc/{doc_id}/"
    print(f"Scraping full case text for ID {doc_id} ({case_url})...")
    
    try:
        case_doc = app.scrape(case_url)
        # Convert Pydantic object to dict if it isn't one already
        if not isinstance(case_doc, dict):
            case_doc = case_doc.model_dump() if hasattr(case_doc, 'model_dump') else vars(case_doc)
            
        case_md = case_doc.get('markdown', '')
        case_meta = case_doc.get('metadata', {})
        
        title = case_meta.get('title', f"Indian Kanoon Case {doc_id}")
        title = title.replace(' - Indian Kanoon', '').strip()
        
        summary = case_md[:350].replace('\n', ' ').strip() + "..."
        full_text = case_md
        
        new_cases.append({
            "case_id": doc_id,
            "title": title,
            "summary": summary,
            "full_text": full_text[:4000], 
            "source_url": case_url
        })
        added += 1
    except Exception as e:
        print(f"  Error extracting full case context: {e}")

existing_data.extend(new_cases)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, indent=4)

print(f"SUCCESS: Expanded dataset saved to {output_file}")
print(f"ADDED: {len(new_cases)} new high-quality records. TOTAL RECORDS: {len(existing_data)}")
