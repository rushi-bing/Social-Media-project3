import os
import threading
from queue import Queue
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from collections import defaultdict
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("skill_analysis_4chan", log_file='logs/skill_analysis_4chan.log', max_bytes=10*1024*1024, backup_count=5)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/jobMarketDB")
DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "jobMarketDB")
COLLECTIONS = ["4chan_posts_comments", "4chan_politics_comments"]

# Multithreading Configuration
NUM_THREADS = 5
BATCH_SIZE = 1000  # Number of records per batch

def load_skill_categories(file_path="skills.txt"):
    """Load skill categories from a text file."""
    skill_categories = {}
    with open(file_path, "r") as file:
        current_category = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                current_category = line.replace(":", "").strip()
                skill_categories[current_category] = []
            elif current_category:
                skill_categories[current_category].append(line.lower())
    return skill_categories

def extract_skills(text, skill_categories):
    """Extract skills from text using skill categories."""
    text = text.lower()
    skill_counts = defaultdict(int)
    for category, keywords in skill_categories.items():
        for keyword in keywords:
            if keyword in text:
                skill_counts[category] += 1
    return skill_counts

def process_batch(batch, skill_categories, weekly_results, lock):
    """Process a batch of records and calculate weekly skill demand."""
    local_results = defaultdict(lambda: defaultdict(int))
    for record in batch:
        try:
            text = record.get("comment", "") + " " + record.get("thread_subject", "")
            date = pd.to_datetime(record.get("timestamp"), unit="s").to_period("W")
            skills = extract_skills(text, skill_categories)
            for category, count in skills.items():
                local_results[date][category] += count
        except Exception as e:
            logger.error(f"Error processing record: {e}")

    # Update shared results
    with lock:
        for week, categories in local_results.items():
            for category, count in categories.items():
                weekly_results[week][category] += count

def worker(queue, skill_categories, weekly_results, lock):
    """Thread worker function to process batches from the queue."""
    while True:
        batch = queue.get()
        if batch is None:
            break
        process_batch(batch, skill_categories, weekly_results, lock)
        queue.task_done()

def fetch_and_process_data(collection, skill_categories, weekly_results, lock):
    """Fetch data from MongoDB in batches and process it."""
    queue = Queue()
    threads = []

    # Start worker threads
    for _ in range(NUM_THREADS):
        thread = threading.Thread(target=worker, args=(queue, skill_categories, weekly_results, lock))
        thread.start()
        threads.append(thread)

    # Fetch records in batches and add to the queue
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

def generate_heatmap(weekly_results, output_file="skill_demand_heatmap_4chan.png"):
    """Generate and save a heatmap of skill demand."""
    df = pd.DataFrame.from_dict(weekly_results, orient="index").fillna(0)
    df.index = df.index.to_timestamp()
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.T, cmap="YlGnBu", annot=False, fmt="d", cbar=True)
    plt.title("4chan Skill Demand Heatmap (Weekly)")
    plt.xlabel("Week")
    plt.ylabel("Skill Categories")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    logger.info(f"Skill demand heatmap saved as {output_file}")

def main():
    """Main function for skill demand analysis."""
    try:
        # Load skill categories
        skill_categories = load_skill_categories("skills.txt")

        # MongoDB connection
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        weekly_results = defaultdict(lambda: defaultdict(int))
        lock = threading.Lock()

        # Process each collection
        for collection_name in COLLECTIONS:
            logger.info(f"Processing collection: {collection_name}")
            collection = db[collection_name]
            fetch_and_process_data(collection, skill_categories, weekly_results, lock)

        # Generate heatmap
        if weekly_results:
            generate_heatmap(weekly_results)
        else:
            logger.warning("No data available for heatmap generation.")
    except Exception as e:
        logger.error(f"Skill demand analysis failed: {e}")

if __name__ == "__main__":
    main()
