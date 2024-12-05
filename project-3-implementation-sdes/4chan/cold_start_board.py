import os
from pyfaktory import Client, Consumer, Job, Producer
import sys
from logger_setup import setup_logger

# Set up the logger at the module level
logger = setup_logger("4chan board cold start", log_file='logs/crawler_cold_start.log', max_bytes=10*1024*1024, backup_count=5)

FAKTORY_SERVER_URL = os.environ.get("FAKTORY_SERVER_URL")

if __name__ == "__main__":
    board = sys.argv[1]
    print(f"Cold starting catalog crawl for board {board}")

    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        job = Job(jobtype="crawl-catalog", args=(board,[]), queue="crawl-catalog")
        producer.push(job)
