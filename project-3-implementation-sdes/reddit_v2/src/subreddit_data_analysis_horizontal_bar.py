import os
import threading
from queue import Queue
from pymongo import MongoClient
import matplotlib.pyplot as plt
from logger_setup import setup_logger

# Logger setup
logger = setup_logger("subreddit_data_analysis", log_file="logs/subreddit_analysis.log", max_bytes=10*1024*1024, backup_count=5)

# MongoDB Configuration
MONGODB_URI = "mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB"
COMMENTS_COLLECTION = "reddit_comments"
POSTS_COLLECTION = "reddit_posts"

# List of subreddits to analyze
SUBREDDITS = ["technology", "csMajors", "cscareerquestions", "programming", "jobs", "recruitinghell"]

# Multithreading configuration
NUM_THREADS = 6
BATCH_SIZE = 500


# Worker function for counting records in batches
def count_subreddit_data(queue, subreddit_counts, lock):
    while True:
        task = queue.get()
        if task is None:
            break

        collection, subreddit = task
        try:
            # Count records for the given subreddit in the specified collection
            client = MongoClient(MONGODB_URI)
            db = client["jobMarketDB"]
            count = db[collection].count_documents({"subreddit": subreddit})

            # Update the global count safely using a lock
            with lock:
                subreddit_counts[subreddit] += count
            logger.info(f"Counted {count} records for subreddit '{subreddit}' in collection '{collection}'.")

        except Exception as e:
            logger.error(f"Error counting data for subreddit '{subreddit}' in collection '{collection}': {e}")
        finally:
            queue.task_done()


# Function to fetch data and process it in parallel
def fetch_and_count_data():
    try:
        subreddit_counts = {subreddit: 0 for subreddit in SUBREDDITS}
        queue = Queue()
        lock = threading.Lock()
        threads = []

        # Create and start threads
        for _ in range(NUM_THREADS):
            thread = threading.Thread(target=count_subreddit_data, args=(queue, subreddit_counts, lock))
            thread.start()
            threads.append(thread)

        # Enqueue tasks for both collections
        for subreddit in SUBREDDITS:
            queue.put((COMMENTS_COLLECTION, subreddit))
            queue.put((POSTS_COLLECTION, subreddit))

        # Wait for the queue to empty
        queue.join()

        # Signal threads to stop
        for _ in range(NUM_THREADS):
            queue.put(None)
        for thread in threads:
            thread.join()

        return subreddit_counts

    except Exception as e:
        logger.error(f"Error in fetch_and_count_data: {e}")
        return {}


# Function to generate horizontal bar chart
def plot_horizontal_bar_chart(data):
    try:
        subreddits = list(data.keys())
        counts = list(data.values())

        # Create a horizontal bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(subreddits, counts, color="skyblue")
        plt.xlabel("Count of Records")
        plt.ylabel("Subreddits")
        plt.title("Data Distribution Across Subreddits")
        plt.tight_layout()

        # Save the chart as an image
        script_directory = os.path.dirname(os.path.abspath(__file__))
        chart_path = os.path.join(script_directory, "subreddit_data_distribution_multithreaded.png")
        plt.savefig(chart_path)
        logger.info(f"Horizontal bar chart saved at: {chart_path}")
        plt.show()

    except Exception as e:
        logger.error(f"Error while plotting horizontal bar chart: {e}")


if __name__ == "__main__":
    logger.info("Starting subreddit data analysis with multithreading...")
    subreddit_data = fetch_and_count_data()

    if subreddit_data:
        logger.info(f"Subreddit data counts: {subreddit_data}")
        plot_horizontal_bar_chart(subreddit_data)
    else:
        logger.warning("No data found for analysis.")
