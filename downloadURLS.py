import pandas as pd

# Generate quarterly URLs from 2012 Q1 to 2024 Q3
years = range(2012, 2025)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

urls = []
for year in years:
    for quarter in quarters:
        if year == 2024 and quarter == 'Q4':  # Stop at 2024 Q3
            break
        url = f"https://fis.fda.gov/content/Exports/faers_ascii_{year}{quarter}.zip"
        urls.append({
            'Year': year,
            'Quarter': quarter,
            'URL': url,
            'Filename': f"faers_ascii_{year}{quarter}.zip"
        })

df_urls = pd.DataFrame(urls)
print(f"Total quarters to download: {len(df_urls)}")
print(df_urls.head(10))

# Save to CSV for tracking
df_urls.to_csv('faers_download_urls.csv', index=False)