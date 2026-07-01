import json
import wikipedia
import sys
from pathlib import Path

seed_cases_path = str(Path(__file__).resolve().parent.parent / "data" / "seed_cases.json")

def fix_seed_cases():
    print("Fixing seed_cases.json with Wikipedia data...")
    with open(seed_cases_path, "r", encoding="utf-8") as f:
        seed_cases = json.load(f)
        
    for c in seed_cases:
        if len(c.get("full_text", "")) < 2000:
            title = c['title']
            print(f"Searching Wikipedia for {title}...")
            try:
                # Get page text
                search_results = wikipedia.search(title)
                if search_results:
                    page = wikipedia.page(search_results[0], auto_suggest=False)
                    text = page.content
                    if len(text) > 2000:
                        c["full_text"] = text[:4000] # limit to 4000 for consistency
                        print(f"  -> Found on Wiki! ({search_results[0]})")
                    else:
                        print(f"  -> Page too short.")
                else:
                    print(f"  -> No Wikipedia results.")
            except Exception as e:
                print(f"  -> Wikipedia error: {e}")
                
    with open(seed_cases_path, "w", encoding="utf-8") as f:
        json.dump(seed_cases, f, indent=4)
    print("seed_cases.json fixed!")

if __name__ == "__main__":
    fix_seed_cases()
