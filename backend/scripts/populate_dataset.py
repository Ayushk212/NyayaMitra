import json
import concurrent.futures
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-de9ac1a674d44cb2a8463b02ab2f9459")
dataset_path = str(Path(__file__).resolve().parent.parent / "dataset.json")

def scrape_case(case_data):
    case_url = case_data.get("source_url")
    if not case_url:
        return case_data
    try:
        case_doc = app.scrape(case_url)
        if not isinstance(case_doc, dict):
            case_doc = case_doc.model_dump() if hasattr(case_doc, 'model_dump') else vars(case_doc)
        
        case_md = case_doc.get('markdown', '')
        if case_md:
            case_data["summary"] = case_md[:350].replace('\n', ' ').strip() + "..."
            case_data["full_text"] = case_md[:4000]
        return case_data
    except Exception as e:
        print(f"Error scraping {case_url}: {e}")
        return case_data

def main():
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    missing_cases = [c for c in dataset if not c.get("full_text")]
    print(f"Found {len(missing_cases)} cases missing full_text out of {len(dataset)} total.")

    if not missing_cases:
        print("All cases already populated. Exiting.")
        return

    print("Starting scraping concurrently with 10 workers...")
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scrape_case, c): c for c in missing_cases}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(missing_cases)} cases...")
                with open(dataset_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=4)

    # Final save
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"Scraping completed! Fully populated dataset saved to {dataset_path}")

if __name__ == "__main__":
    main()
