import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("reddit_analysis", log_file='logs/reddit_analysis.log', max_bytes=10*1024*1024, backup_count=5)

def process_chunk(chunk):
    processed_data = []
    for record in chunk:
        processed_data.append({
            "date": pd.to_datetime(record["utc"]).date(),
            "is_normal": record.get("toxicity", {}).get("class") != "flag"
        })
    logger.info("Chunk processed")
    return processed_data

def chunk_generator(cursor, chunk_size):
    chunk = []
    for record in cursor:
        chunk.append(record)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

client = pymongo.MongoClient("mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB")
db = client['jobMarketDB']
collection = db['reddit_posts']

cursor = collection.find(
    {"subreddit": {"$ne": "politics"}},
    {"utc": 1, "toxicity.class": 1}
)

chunk_size = 10000
data = []

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunk_generator(cursor, chunk_size)]
    for future in futures:
        data.extend(future.result())

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

aggregated = df.groupby("date").is_normal.value_counts().unstack(fill_value=0)
aggregated.rename(columns={True: "Normal", False: "Flagged"}, inplace=True)

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Plot "Flagged Comments"
axes[0].plot(aggregated.index, aggregated["Flagged"], color="red", label="Flagged Comments")
axes[0].set_xlabel("Date (ISO 8601)")
axes[0].set_ylabel("Comment Count")
axes[0].set_title("Daily Flagged Comment Counts")
axes[0].legend()
axes[0].grid(True)

# Plot "Normal Comments"
axes[1].plot(aggregated.index, aggregated["Normal"], color="blue", label="Normal Comments")
axes[1].set_xlabel("Date (ISO 8601)")
axes[1].set_ylabel("Comment Count")
axes[1].set_title("Daily Normal Comment Counts")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('Toxicity_posts_subplots.png')
plt.show()

print("Plots saved as 'Toxicity_posts_subplots.png'.")
