import os
from dotenv import load_dotenv
import threading
from queue import Queue
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from logger_setup import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("4chan_analysis", log_file='logs/4chan_analysis.log', max_bytes=10*1024*1024, backup_count=5)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME")
COLLECTIONS = [os.getenv("MONGODB_DB_COLLECTION_NAME")]

# Multithreading Configuration
NUM_THREADS = 5
BATCH_SIZE = 1000  # Number of records per batch

def connect_to_mongodb():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        return db
    except pymongo.errors.ConnectionError as ce:
        logger.error(f"MongoDB connection error: {ce}")
        raise

def process_batch(batch, results, lock):
    """Process a batch of records and calculate daily toxicity counts."""
    local_results = defaultdict(lambda: {"flag": 0, "neutral": 0})

    for record in batch:
        try:
            timestamp = record.get("timestamp")
            toxicity_class = record.get("toxicity", {}).get("class", "neutral")

            if not timestamp or toxicity_class not in ["flag", "neutral"]:
                continue

            # Convert timestamp to date string
            date = pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d")

            # Update local results
            local_results[date][toxicity_class] += 1
        except Exception as e:
            logger.error(f"Error processing record: {e}")

    # Update shared results with thread safety
    with lock:
        for date, counts in local_results.items():
            results[date]["flag"] += counts["flag"]
            results[date]["neutral"] += counts["neutral"]

def fetch_toxicity_data_multithreaded(db, collection_name):
    """
    Fetch and process daily toxicity counts using multithreading.

    Parameters:
        db: MongoDB database instance.
        collection_name: Name of the MongoDB collection.

    Returns:
        pd.DataFrame: A DataFrame containing daily counts of flagged and neutral posts/comments.
    """
    try:
        collection = db[collection_name]
        results = defaultdict(lambda: {"flag": 0, "neutral": 0})
        lock = threading.Lock()
        queue = Queue()
        threads = []

        # Define worker function
        def worker():
            while True:
                batch = queue.get()
                if batch is None:
                    break
                process_batch(batch, results, lock)
                queue.task_done()

        # Start worker threads
        for _ in range(NUM_THREADS):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        # Fetch records in batches
        last_id = None
        while True:
            query = {}
            if last_id:
                query["_id"] = {"$gt": last_id}

            batch = list(collection.find(query).sort("_id").limit(BATCH_SIZE))
            if not batch:
                break

            queue.put(batch)
            last_id = batch[-1]["_id"]

        # Wait for all tasks to complete
        queue.join()

        # Stop worker threads
        for _ in range(NUM_THREADS):
            queue.put(None)
        for thread in threads:
            thread.join()

        # Convert results to DataFrame
        data = [
            {"date": date, "toxicity_class": "flag", "count": counts["flag"]}
            for date, counts in results.items()
        ] + [
            {"date": date, "toxicity_class": "neutral", "count": counts["neutral"]}
            for date, counts in results.items()
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for collection {collection_name}: {e}")
        return pd.DataFrame()

def process_and_plot_data(dataframes, output_file="img/4chan_toxicity_trend.png"):
    """
    Process and plot daily toxicity data.

    Parameters:
        dataframes (list): List of DataFrames for all collections.
        output_file (str): File name to save the plot.
    """
    try:
        # Combine data from all collections
        combined_data = pd.concat(dataframes, ignore_index=True)

        # Pivot data to get daily counts for each toxicity class
        pivot_table = combined_data.pivot_table(
            index="date",
            columns="toxicity_class",
            values="count",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Sort data by date
        pivot_table["date"] = pd.to_datetime(pivot_table["date"])
        pivot_table.sort_values("date", inplace=True)

        # Plot data
        plt.figure(figsize=(12, 6))
        for toxicity_class in pivot_table.columns[1:]:
            plt.plot(
                pivot_table["date"],
                pivot_table[toxicity_class],
                label=toxicity_class.capitalize()
            )

        plt.title("Daily Counts of Flagged and Normal Posts/Comments on 4chan")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

        logger.info(f"Toxicity trend plot saved as {output_file}.")
    except Exception as e:
        logger.error(f"Error processing and plotting data: {e}")

def main():
    """Main function to analyze and visualize daily toxicity data."""
    try:
        db = connect_to_mongodb()
        dataframes = []

        # Fetch toxicity data for each collection
        for collection_name in COLLECTIONS:
            logger.info(f"Fetching data for collection: {collection_name}")
            df = fetch_toxicity_data_multithreaded(db, collection_name)
            if not df.empty:
                dataframes.append(df)

        # Generate and save the plot
        if dataframes:
            process_and_plot_data(dataframes)
        else:
            logger.warning("No data available for plotting.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
