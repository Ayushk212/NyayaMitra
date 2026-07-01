from firecrawl import Firecrawl
import json

app = Firecrawl(api_key="fc-de9ac1a674d44cb2a8463b02ab2f9459")

# Scrape a website:
print("Scraping firecrawl.dev...")
result = app.scrape_url('https://firecrawl.dev')

print("Result keys:", result.keys() if isinstance(result, dict) else type(result))
if isinstance(result, dict) and 'markdown' in result:
    print("\nMarkdown excerpt:")
    print(result['markdown'][:500] + "...\n[TRUNCATED]")
else:
    print("\nContent excerpt:")
    print(str(result)[:500])
