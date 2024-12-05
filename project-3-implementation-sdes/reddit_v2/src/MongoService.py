from pymongo import MongoClient, errors
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("reddit", log_file='logs/reddit.log', max_bytes=10*1024*1024, backup_count=5)

def get_mongo_client():
    """Initialize MongoDB client."""
    try:
        client = MongoClient("mongodb://vchoudhary:password@127.0.0.1:27017/jobMarketDB")
        return client
    except errors.ConnectionError as ce:
        logger.error(f"MongoDB connection error: {ce}")
        raise

def insert_to_mongodb(records, collection):
    """Inserts or replaces records in MongoDB using upsert."""
    if not records:
        return "No records to insert."

    try:
        mongo_client = get_mongo_client()
        db = mongo_client['jobMarketDB']
        dbcollection = db[collection]
        for record in records:
            dbcollection.replace_one(
                {"_id": record["_id"]},
                record,
                upsert=True
            )

        return "Success"

    except errors.PyMongoError as pe:
        logger.exception(f"MongoDB operation failed: {pe}")
        return f"Failure: {str(pe)}"

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return f"Failure: {str(e)}"
