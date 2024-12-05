import os
import re
import html
from pymongo import MongoClient, errors, UpdateOne
from concurrent.futures import ThreadPoolExecutor
from logger_setup import setup_logger
from ToxicityApiClient import analyze_toxicity
from dotenv import load_dotenv

# Setup logger
logger = setup_logger("4chan_backfill", log_file='logs/4chan_backfill.log', max_bytes=10*1024*1024, backup_count=5)

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "jobMarketDB")
COLLECTIONS = [
    os.getenv("MONGODB_DB_COLLECTION_NAME", "4chan_posts_comments"),
]
BATCH_SIZE = 300  # Number of records per batch
WORKER_COUNT = 5  # Number of parallel threads

# Precompiled regex patterns
quote_pattern = re.compile(r"&gt;&gt;\d+")
tag_pattern = re.compile(r"<[^>]*>")

def preprocess_comment(comment):
    """
    Preprocess the comment by decoding HTML entities and removing HTML tags.

    Parameters:
        comment (str): The raw HTML-encoded comment.

    Returns:
        str: The cleaned, plain-text comment.
    """
    try:
        # Decode HTML entities (e.g., &gt; to >)
        decoded_comment = html.unescape(comment)

        # Replace quoted links (e.g., >>102937844) with a placeholder
        decoded_comment = quote_pattern.sub("[quote]", decoded_comment)

        # Remove HTML tags using a regular expression
        cleaned_comment = tag_pattern.sub("", decoded_comment).strip()

        return cleaned_comment
    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return comment  # Return the original comment if preprocessing fails

def connect_to_mongodb():
    """Connect to MongoDB."""
    try:
        client = MongoClient(MONGODB_URI, maxPoolSize=50)  # Enable connection pooling
        db = client[DATABASE_NAME]
        return db
    except errors.PyMongoError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def process_chunk(records, collection):
    """
    Process a chunk of records by analyzing toxicity and updating the database.

    Parameters:
        records (list): List of MongoDB records.
        collection: The MongoDB collection object.
    """
    default_value={"class": "neutral", "confidence": 0.0}
    try:
        bulk_updates = []
        for record in records:
            try:
                comment = record.get("comment", "")
                if not comment:
                    logger.warning(f"No comment found for record ID: {record['_id']}")
                    continue

                # Preprocess the comment
                cleaned_comment = preprocess_comment(comment)

                # Analyze toxicity
                toxicity_result = analyze_toxicity(cleaned_comment)
                
                # Use default value if toxicity analysis fails
                if not toxicity_result:
                    logger.warning(f"Toxicity analysis failed for record ID: {record['_id']}")
                    toxicity_result = default_value


                # Prepare bulk update
                if toxicity_result:
                    bulk_updates.append(
                        UpdateOne(
                            {"_id": record["_id"]},
                            {"$set": {"toxicity": toxicity_result}}
                        )
                    )
            except Exception as e:
                logger.error(f"Error processing record ID {record['_id']}: {e}")

        # Execute bulk update
        if bulk_updates:
            collection.bulk_write(bulk_updates)
            logger.info(f"Processed and updated {len(bulk_updates)} records.")
    except Exception as e:
        logger.error(f"Error in processing chunk: {e}")

def backfill_toxicity_analysis(db):
    """Backfill toxicity analysis for 4chan collections."""
    for collection_name in COLLECTIONS:
        collection = db[collection_name]
        logger.info(f"Starting backfill for collection: {collection_name}")

        # Create index on `toxicity` if it doesn't exist
        collection.create_index("toxicity")

        # Process records in parallel
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            last_id = None
            while True:
                try:
                    # Fetch a batch of records
                    query = {"toxicity": {"$exists": False}}
                    if last_id:
                        query["_id"] = {"$gt": last_id}

                    records = list(collection.find(query).sort("_id").limit(BATCH_SIZE))
                    if not records:
                        break  # Exit if no more records

                    # Submit batch to the executor
                    executor.submit(process_chunk, records, collection)

                    # Update last_id for pagination
                    last_id = records[-1]["_id"]
                except Exception as e:
                    logger.error(f"Error during batch retrieval: {e}")

        logger.info(f"Completed backfill for collection: {collection_name}")

def main():
    """Main function to backfill toxicity analysis."""
    try:
        db = connect_to_mongodb()
        backfill_toxicity_analysis(db)
    except Exception as e:
        logger.error(f"Backfill process failed: {e}")

if __name__ == "__main__":
    main()
