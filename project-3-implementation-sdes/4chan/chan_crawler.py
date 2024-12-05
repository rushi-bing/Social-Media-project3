import os
import time
import datetime
import re
import html

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from pyfaktory import Client, Consumer, Job, Producer
from ToxicityApiClient import analyze_toxicity

from chan_client import ChanClient
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("4chan crawler", log_file='logs/4chan_crawler.log', max_bytes=10*1024*1024, backup_count=5)

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
FAKTORY_SERVER_URL = os.environ.get("FAKTORY_SERVER_URL")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME")
MONGODB_DB_COLLECTION_NAME = os.environ.get("MONGODB_DB_COLLECTION_NAME")
POL_COLLECTION_NAME=os.environ.get("POL_COLLECTION_NAME")

# Validate environment variables
if not FAKTORY_SERVER_URL:
    logger.error("FAKTORY_SERVER_URL is not set in environment variables.")
if not MONGODB_URI:
    logger.error("MONGODB_URI is not set in environment variables.")

def thread_numbers_from_catalog(catalog):
    """
    Extract thread numbers from the catalog data.

    Args:
        catalog (list): List of pages containing threads.

    Returns:
        list: List of thread numbers.
    """
    thread_numbers = []
    try:
        if not catalog or not isinstance(catalog, list):
            raise ValueError("Invalid catalog data provided.")

        for page in catalog:
            threads = page.get("threads")
            if threads is None:
                logger.warning("Page missing 'threads' key.")
                continue

            for thread in threads:
                thread_number = thread.get("no")
                if thread_number is not None:
                    thread_numbers.append(thread_number)
                else:
                    logger.warning("Thread missing 'no' key.")
    except Exception as e:
        logger.exception(f"Error in thread_numbers_from_catalog: {e}")
    return thread_numbers

def find_dead_threads(previous_catalog_thread_numbers, current_catalog_thread_numbers):
    """
    Find threads that are present in the current catalog but not in the previous one.

    Args:
        previous_catalog_thread_numbers (list): List of thread numbers from previous catalog.
        current_catalog_thread_numbers (list): List of thread numbers from current catalog.

    Returns:
        set: Set of new thread numbers to crawl.
    """
    dead_thread_numbers = set()
    try:
        dead_thread_numbers = set(previous_catalog_thread_numbers).difference(
        set(current_catalog_thread_numbers)
    )
    except Exception as e:
        logger.exception(f"Error in find_dead_threads: {e}")
    return dead_thread_numbers

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
        decoded_comment = re.sub(r"&gt;&gt;\d+", "[quote]", decoded_comment)

        # Remove HTML tags using a regular expression
        cleaned_comment = re.sub(r"<[^>]*>", "", decoded_comment)

        # Normalize whitespace
        cleaned_comment = re.sub(r"\s+", " ", cleaned_comment).strip()

        return cleaned_comment
    except Exception as e:
        print(f"Error preprocessing comment: {e}")
        return comment  # Return the original comment if preprocessing fails


def crawl_thread(board, thread_number, last_modified=None):
    """
    Crawl a specific thread and insert its posts into MongoDB in bulk,
    ensuring that all non-duplicate records are inserted even if some are duplicates.

    Args:
        board (str): Board name.
        thread_number (int): Thread number.
        last_modified (str): Optional 'If-Modified-Since' header value.
    """
    try:
        # Initialize ChanClient to fetch thread data with If-Modified-Since
        chan_client = ChanClient()
        thread_data = chan_client.get_thread(board, thread_number, last_modified)
        
        # Skip if no new data was found
        if thread_data is None or "posts" not in thread_data:
            logger.info(f"No new data for thread {board}/{thread_number}.")
            return  # Skip processing if thread data is unchanged
    except Exception as e:
        logger.exception(f"Error fetching thread data for {board}/{thread_number}: {e}")
        return  # Cannot proceed without thread data

    try:
        # Validate environment variables
        if not FAKTORY_SERVER_URL:
            logger.error("FAKTORY_SERVER_URL is not set in environment variables.")

        if not MONGODB_URI:
            logger.error("MONGODB_URI is not set in environment variables.")
        
        if not POL_COLLECTION_NAME:
            logger.error("POL_COLLECTION_NAME is not set in environment variables.")

        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE_NAME]  # MongoDB database name
        if board=='pol':
            collection = db[POL_COLLECTION_NAME]  # MongoDB /pol collection name
            collection.create_index([("post_number", 1)],unique=True)
        elif board=='g':
            collection = db[MONGODB_DB_COLLECTION_NAME]# MongoDB /g collection name
            
            # Create a compound unique index on board, thread_number, and post_number
            collection.create_index(
                [("board", 1), ("thread_number", 1), ("post_number", 1)],
                unique=True
            )
        else:
            logger.error("MONGODB_DB_COLLECTION_NAME not setup")
            return # Cannot proceed without collection name

    except Exception as e:
        logger.exception(f"Error connecting to MongoDB: {e}")
        return  # Cannot proceed without database connection

    try:
        # Prepare a list to collect documents for bulk insertion
        documents_to_insert = []

        # Process each post and add it to the list
        if board=='pol':
            for post in thread_data.get("posts", []):
                post_number_value = post.get("no", "")
                parent_thread = post.get("resto", 0)
                post_date_time = post.get("now", "")
                timestamp = post.get("time", "")
                documents_to_insert.append(
                    {
                        "post_number": post_number_value,
                        "post_date_time": post_date_time,
                        "timestamp": timestamp,
                        "parent_thread": parent_thread,
                     }
                )
        else:
            for post in thread_data.get("posts", []):
                # Extract post data with default values
                crawler_board = board or ""
                thread_number_value = thread_number or ""
                post_number_value = post.get("no", "")
                post_date_time = post.get("now", "")
                semantic_url = post.get("semantic_url", "")
                author_name = post.get("name", "Anonymous")
                thread_subject = post.get("sub", "")
                timestamp = post.get("time", "")
                parent_thread = post.get("resto", 0)
                is_archived = post.get("archived", 0)
                thread_replies = post.get("replies", 0)

                comment = post.get("com", "")
                cleaned_comment = preprocess_comment(comment)

                # Build the document to insert
                document = {
                    "board": crawler_board,
                    "thread_number": thread_number_value,
                    "post_number": post_number_value,
                    "post_date_time": post_date_time,
                    "author_name": author_name,
                    "comment": comment,
                    "timestamp": timestamp,
                    "parent_thread": parent_thread,
                }

                if parent_thread == 0:
                    # Original thread post
                    document.update({
                        "semantic_url": semantic_url,
                        "thread_subject": thread_subject,
                        "is_archived": is_archived,
                        "thread_replies": thread_replies,
                    })

                # Perform toxicity analysis
                if cleaned_comment:
                    toxicity_result = analyze_toxicity(cleaned_comment)
                    if toxicity_result:
                        document["toxicity"] = toxicity_result

                # Add the document to the list
                documents_to_insert.append(document)

        # Perform bulk insertion with ordered=False to continue on errors
        try:
            if documents_to_insert:
                result = collection.insert_many(documents_to_insert, ordered=False)
                inserted_count = len(result.inserted_ids)
                logger.info(f"Inserted {inserted_count} /{board} documents into MongoDB for thread {thread_number}.")
            else:
                logger.info(f"No new /{board} documents to insert for thread {thread_number}.")
        except BulkWriteError as bwe:
            # Handle duplicate key errors and log them
            write_errors = bwe.details.get('writeErrors', [])
            inserted_count = bwe.details.get('nInserted', 0)
            logger.info(f"Inserted {inserted_count} /{board} new documents into MongoDB for thread {thread_number}.")

            for error in write_errors:
                if error.get('code') == 11000:
                    # Duplicate key error
                    post_number = error['op'].get('post_number', 'Unknown')
                    logger.info(f"DUPLICATE: /{board} Post number {post_number} already exists. Skipping insertion.")
                else:
                    logger.error(f"Write error board: /{board}: {error}")
        except Exception as e:
            logger.exception(f"Error during bulk insert board: /{board}: {e}")
            return
    except Exception as e:
        logger.exception(f"Error processing board: /{board} posts in thread {thread_number}: {e}")
        return
    finally:
        # Ensure MongoDB connection is closed
        try:
            client.close()
        except Exception as e:
            logger.exception(f"Error closing MongoDB connection: {e}")

    # Delay 10 seconds between thread crawls
    time.sleep(10)

def crawl_catalog(board, previous_catalog_thread_numbers=[]):
    """
    Crawl the catalog for a given board and schedule thread crawls.

    Args:
        board (str): Board name.
        previous_catalog_thread_numbers (list): List of thread numbers from previous catalog.
    """
    max_retries = 3  # Maximum number of retries
    delay_between_retries = 5  # Delay in seconds between retries

    # Attempt to fetch catalog data with retries
    for attempt in range(1, max_retries + 1):
        try:
            # Initialize ChanClient to fetch catalog data
            chan_client = ChanClient()
            current_catalog = chan_client.get_catalog(board)
            current_catalog_thread_numbers = thread_numbers_from_catalog(current_catalog)
            logger.info(f"Successfully fetched catalog data on attempt {attempt}.")
            break  # Exit the loop if successful
        except Exception as e:
            logger.exception(f"Error fetching catalog for board {board} on attempt {attempt}: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                logger.error(f"Failed to fetch catalog after {max_retries} attempts. Exiting function.")
                return  # Cannot proceed without catalog data

    try:
        # Find dead threads that need to be crawled
        dead_threads = find_dead_threads(
            previous_catalog_thread_numbers, current_catalog_thread_numbers
        )
        logger.info(f"Threads to crawl: {dead_threads}")
    except Exception as e:
        logger.exception(f"Error finding threads to crawl: {e}")
        dead_threads = set()

    # Schedule crawl-thread jobs for each new thread
    try:
        if dead_threads:
            crawl_thread_jobs = []
            with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
                producer = Producer(client=client)
                for dead_thread in dead_threads:
                    job = Job(
                        jobtype="crawl-thread",
                        args=(board, dead_thread),
                        queue="crawl-thread"
                    )
                    crawl_thread_jobs.append(job)
                producer.push_bulk(crawl_thread_jobs)
                logger.info(f"Scheduled crawl-{board}-thread jobs for threads: {dead_threads}")
    except Exception as e:
        logger.exception(f"Error scheduling crawl-{board}-thread jobs: {e}")

    # Schedule the next catalog crawl
    try:
        with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
            producer = Producer(client=client)
            # Calculate the time for the next crawl
            run_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
            run_at_local = run_at.astimezone()  # Converts to local time zone
            run_at_iso_local = run_at_local.isoformat()
            logger.info(f"***Scheduling next /{board} crawl at: {run_at_iso_local} ***")
            job = Job(
                jobtype="crawl-catalog",
                args=(board, current_catalog_thread_numbers),
                queue="crawl-catalog",
                at=run_at.isoformat(),
            )
            producer.push(job)
    except Exception as e:
        logger.exception(f"Error scheduling next crawl-catalog job: {e}")

def main():
    """
    Main function to start the consumer and handle exceptions to keep the application running.
    """
    while True:
        try:
            with Client(faktory_url=FAKTORY_SERVER_URL, role="consumer") as client:
                consumer = Consumer(
                    client=client,
                    queues=["crawl-catalog", "crawl-thread"],
                    concurrency=5
                )
                # Register job types with their corresponding functions
                consumer.register("crawl-catalog", crawl_catalog)
                consumer.register("crawl-thread", crawl_thread)
                logger.info("Starting consumer to process jobs.")
                # Start consuming jobs
                consumer.run()
        except KeyboardInterrupt:
            # Graceful shutdown on Ctrl+C
            logger.exception("Keyboard interrupt received. Shutting down.")
            break
        except Exception as e:
            # Log the exception and restart the loop after a delay
            logger.exception(f"Error in main execution: {e}")
            logger.info("Restarting the consumer after a 10 sec short delay.")
            time.sleep(10)  # Delay before restarting to prevent rapid restarts

if __name__ == "__main__":
    """Main function to start the consumer and handle exceptions to keep the application running."""
    main()