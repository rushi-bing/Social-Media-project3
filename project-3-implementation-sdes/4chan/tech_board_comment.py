import os
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

MONGODB_URI_4CHAN = "mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB"
MONGODB_DATABASE_NAME_4CHAN = "jobMarketDB"
COLLECTION_NAME_4CHAN = "4chan_posts_comments"

client_4chan = MongoClient(MONGODB_URI_4CHAN)
db_4chan = client_4chan[MONGODB_DATABASE_NAME_4CHAN]
collection_4chan = db_4chan[COLLECTION_NAME_4CHAN]

MONGODB_URI_REDDIT = "mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB"
MONGODB_DATABASE_NAME_REDDIT = "jobMarketDB"
COLLECTION_NAME_REDDIT = "reddit_comments"

client_reddit = MongoClient(MONGODB_URI_REDDIT)
db_reddit = client_reddit[MONGODB_DATABASE_NAME_REDDIT]
collection_reddit = db_reddit[COLLECTION_NAME_REDDIT]

# Define date range
start_date = "2024-10-18"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

# Generate a complete hourly date range with timezone awareness
full_range = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')

query_4chan = {
    "timestamp": {
        "$gte": int(pd.Timestamp(start_date).timestamp()),
        "$lte": int(pd.Timestamp(end_date).timestamp())
    }
}
data_4chan = list(collection_4chan.find(query_4chan))

# Convert 4chan data into a DataFrame
df_4chan = pd.DataFrame(data_4chan)

# Convert 'timestamp' to datetime with timezone awareness
df_4chan['datetime'] = pd.to_datetime(df_4chan['timestamp'], unit='s', utc=True)

# Resample 4chan data hourly
hourly_counts_4chan = df_4chan.resample('H', on='datetime').size()

hourly_counts_4chan = hourly_counts_4chan.reindex(full_range, fill_value=0)

start_date_str = f"{start_date}T00:00:00Z"
end_date_str = f"{end_date}T23:59:59Z"

query_reddit = {
    "$and": [
        {
            "utc": {
                "$gte": start_date_str,
                "$lte": end_date_str
            }
        },
        {
            "subreddit": {"$ne": "politics"}
        }
    ]
}
data_reddit = list(collection_reddit.find(query_reddit))

# Convert Reddit data into a DataFrame
df_reddit = pd.DataFrame(data_reddit)

# Ensure 'utc' field exists and is not empty
if 'utc' in df_reddit.columns and not df_reddit['utc'].isnull().all():
    # Convert 'utc' to datetime
    df_reddit['datetime'] = pd.to_datetime(df_reddit['utc'])
else:
    print("No valid 'utc' field found in Reddit data.")
    df_reddit['datetime'] = pd.NaT

df_reddit = df_reddit.dropna(subset=['datetime'])

# Resample Reddit data hourly
hourly_counts_reddit = df_reddit.resample('H', on='datetime').size()

hourly_counts_reddit = hourly_counts_reddit.reindex(full_range, fill_value=0)

plt.figure(figsize=(30, 10))

# Plot 4chan data
plt.plot(hourly_counts_4chan.index, hourly_counts_4chan.values, label='4chan', marker='.', markersize=5, linestyle='-', linewidth=1)

# Plot Reddit data
plt.plot(hourly_counts_reddit.index, hourly_counts_reddit.values, label='Reddit', marker='.', markersize=5, linestyle='-', linewidth=1)

plt.title("Number of Comments per Hour on 4chan and Reddit", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Number of Comments", fontsize=14)
plt.legend()

# Set x-axis ticks for each day
tick_positions = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
tick_labels = [tick.strftime('%Y-%m-%d') for tick in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.grid()
plt.tight_layout()

output_file = "img/comments_per_hour_4chan_reddit.png"
plt.savefig(output_file)
print(f"Plot saved as {output_file}")