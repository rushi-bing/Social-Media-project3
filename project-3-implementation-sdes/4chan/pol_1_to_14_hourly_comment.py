import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME")
POL_COLLECTION_NAME = os.environ.get("POL_COLLECTION_NAME")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE_NAME]
collection = db[POL_COLLECTION_NAME]

# Define date range
start_date = "2024-11-01"
end_date = "2024-11-15"

# Query MongoDB for the data
query = {
    "timestamp": {
        "$gte": int(pd.Timestamp(start_date).timestamp()),
        "$lte": int(pd.Timestamp(end_date).timestamp())
    }
}
data = list(collection.find(query))

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Convert timestamp to datetime for processing
df['post_date_time'] = pd.to_datetime(df['timestamp'], unit='s')

# Count comments per hour
hourly_counts = df.resample('H', on='post_date_time').size()

# Generate a complete hourly range
full_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Reindex to include all hours, filling missing values with 0
hourly_counts = hourly_counts.reindex(full_range, fill_value=0)

# Plot the data and save as PNG
plt.figure(figsize=(14, 7))
plt.plot(hourly_counts.index, hourly_counts.values, marker='o')

# Customize x-axis to show all days
plt.title("Number of Comments per Hour on 4chan's /pol/ Board (Nov 1-14, 2024)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Number of Comments", fontsize=14)

# Set x-axis ticks for each day
tick_positions = pd.date_range(start=start_date, end=end_date, freq='D')
tick_labels = [tick.strftime('%Y-%m-%d') for tick in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.grid()
plt.tight_layout()

# Save the plot as a PNG file
output_file = "img/comments_per_hour_pol_board.png"
plt.savefig(output_file)
print(f"Plot saved as {output_file}")