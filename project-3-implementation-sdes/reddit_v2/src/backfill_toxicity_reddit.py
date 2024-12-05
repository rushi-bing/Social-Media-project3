from pymongo import MongoClient, errors
from ToxicityApiClient import analyze_toxicity
from logger_setup import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logger
logger = setup_logger("Reddit Backfill", log_file="logs/reddit_backfill.log", max_bytes=10*1024*1024, backup_count=5)

def get_mongo_client():
    """Initialize MongoDB client."""
    try:
        client = MongoClient("mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB")
        return client
    except errors.ConnectionError as ce:
        logger.error(f"MongoDB connection error: {ce}")
        raise

def process_post(post, posts_collection):
    """Process a single Reddit post."""
    try:
        text = post.get("selftext", "").strip()
        title = post.get("title", "").strip()
        _id = post.get("_id")
        if title:
            toxicity_result = analyze_toxicity(title + " " + text)
            posts_collection.update_one({"_id": _id}, {"$set": {"toxicity": toxicity_result}})
            logger.info(f"Updated post {_id} with toxicity: {toxicity_result}")
    except Exception as e:
        logger.error(f"Failed to process post {_id}: {e}")

def process_comment(comment, comments_collection):
    """Process a single Reddit comment."""
    try:
        text = comment.get("body", "").strip()
        _id = comment.get("_id")
        if text:
            toxicity_result = analyze_toxicity(text)
            comments_collection.update_one({"_id": _id}, {"$set": {"toxicity": toxicity_result}})
            logger.info(f"Updated comment {_id} with toxicity: {toxicity_result}")
    except Exception as e:
        logger.error(f"Failed to process comment {_id}: {e}")

def backfill_toxicity_analysis():
    """
    Backfill toxicity analysis for Reddit posts and comments in MongoDB using multithreading.
    """
    try:
        # Connect to MongoDB
        client = get_mongo_client()
        db = client['jobMarketDB']
        posts_collection = db['reddit_posts']
        comments_collection = db['reddit_comments']

        # Backfill toxicity for posts
        logger.info("Starting backfill for Reddit posts...")
        posts = list(posts_collection.find({"toxicity": {"$exists": False}}))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_post, post, posts_collection) for post in posts]
            for future in as_completed(futures):
                # Handling potential exceptions for individual threads
                future.result()

        logger.info("Completed backfill for Reddit posts.")

        # Backfill toxicity for comments
        logger.info("Starting backfill for Reddit comments...")
        comments = list(comments_collection.find({
                        "$and": [
                                {"toxicity": {"$exists": False}},
                                {"subreddit": {"$ne": "politics"}}
                                ]
                            }))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_comment, comment, comments_collection) for comment in comments]
            for future in as_completed(futures):
                # Handling potential exceptions for individual threads
                future.result()

        logger.info("Completed backfill for Reddit comments.")

    except Exception as e:
        logger.error(f"Failed to complete backfill process: {e}")
    finally:
        try:
            client.close()
        except Exception as e:
            logger.error(f"Failed to close MongoDB connection: {e}")

if __name__ == "__main__":
    backfill_toxicity_analysis()
