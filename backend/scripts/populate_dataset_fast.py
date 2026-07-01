import json
import concurrent.futures
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
dataset_path = str(Path(__file__).resolve().parent.parent / "dataset.json")

def scrape_case(case_data):
    case_url = case_data.get("source_url")
    if not case_url:
        return case_data
    try:
        # Use a standard browser User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(case_url, headers=headers, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            judgments_div = soup.find('div', class_='judgments')
            
            if judgments_div:
                text = judgments_div.get_text(separator='\n\n', strip=True)
            else:
                text = soup.body.get_text(separator='\n\n', strip=True) if soup.body else ""
                
            if text:
                case_data["summary"] = text[:350].replace('\n', ' ').strip() + "..."
                case_data["full_text"] = text[:4000]
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

    print("Starting fast scraping concurrently with 30 workers...")
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(scrape_case, c): c for c in missing_cases}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(missing_cases)} cases...")
                with open(dataset_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=4)

    # Final save
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"Scraping completed! Fully populated dataset saved to {dataset_path}")

if __name__ == "__main__":
    main()
