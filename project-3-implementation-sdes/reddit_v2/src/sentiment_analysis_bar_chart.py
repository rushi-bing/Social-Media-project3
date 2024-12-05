import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from logger_setup import setup_logger

# Logger setup
logger = setup_logger("sentiment_analysis", log_file='logs/sentiment_analysis.log', max_bytes=10*1024*1024, backup_count=5)

# MongoDB Configurations
MONGODB_URI = "mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB"
COMMENTS_COLLECTION = "reddit_comments"
POSTS_COLLECTION = "reddit_posts"
BATCH_SIZE = 10000
MAX_WORKERS = 10

# Define the sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

# Define function to process a batch
def process_batch(records, sentiments):
    for record in records:
        try:
            text = record.get("body", "") or record.get("selftext", "") or ""
            if text:
                sentiment = analyze_sentiment(text)
                subreddit = record.get("subreddit", "unknown")
                sentiments.append({"subreddit": subreddit, "sentiment": sentiment})
        except Exception as e:
            logger.error(f"Error processing record {record.get('_id', 'unknown')}: {e}")
    logger.info("Batch processed")


# Query MongoDB and process data
def fetch_and_analyze_sentiments():
    try:
        client = MongoClient(MONGODB_URI)
        db = client["jobMarketDB"]

        # Query comments and posts collections
        collections = [COMMENTS_COLLECTION, POSTS_COLLECTION]
        sentiments = []
        filter_condition = {"subreddit": {"$ne": "politics"}}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for collection_name in collections:
                collection = db[collection_name]

                # Fetch total record count
                total_records = collection.count_documents(filter_condition)
                logger.info(f"Found {total_records} records in {collection_name} for sentiment analysis.")

                # Fetch records in batches
                skip = 0
                while skip < total_records:
                    batch = list(collection.find(filter_condition).skip(skip).limit(BATCH_SIZE))
                    if not batch:
                        break
                    executor.submit(process_batch, batch, sentiments)
                    skip += BATCH_SIZE

        return sentiments

    except Exception as e:
        logger.error(f"Error fetching or analyzing sentiments: {e}")
        return []

# Plot sentiment bar chart
def plot_sentiment_bar_chart(sentiments):
    try:
        # Convert to DataFrame
        df = pd.DataFrame(sentiments)
        sentiment_counts = df.groupby(["subreddit", "sentiment"]).size().unstack(fill_value=0)

        # Plot bar chart
        sentiment_counts.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Sentiment Distribution by Subreddit")
        plt.xlabel("Subreddits")
        plt.ylabel("Counts")
        plt.xticks(rotation=45)

        # Save the graph as an image
        script_directory = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        graph_file_path = os.path.join(script_directory, "sentiment_bar_chart.png")
        plt.savefig(graph_file_path, bbox_inches="tight")
        logger.info(f"Bar graph saved as image at: {graph_file_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error while plotting sentiment bar chart: {e}")

# Main execution
if __name__ == "__main__":
    logger.info("Starting sentiment analysis for all available data")
    sentiments = fetch_and_analyze_sentiments()

    if sentiments:
        plot_sentiment_bar_chart(sentiments)
    else:
        logger.warning("No sentiments to plot.")
