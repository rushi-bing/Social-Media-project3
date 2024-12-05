import logging
from pyfaktory import Client, Producer, Consumer, Job
from datetime import datetime, timedelta, timezone
from FetchPostsJob import get_reddit_posts
from MongoService import insert_to_mongodb
from FetchCommentsJob import fetch_and_process_comments
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("reddit", log_file='logs/reddit.log', max_bytes=10*1024*1024, backup_count=5)

FAKTORY_SERVER_URL = "tcp://:password@localhost:7419"



def get_subreddits_from_file(file_path="subreddits.txt"):
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()
            subreddits = content.split(",")
            subreddits = [sub.strip() for sub in subreddits if sub.strip()]
        return subreddits
    except FileNotFoundError:
        logger.error(f"Error: '{file_path}' not found.")
        return []
    except Exception as e:
        logger.error(f"Error while reading file: {e}")
        return []


def handle_fetch_posts(arg=[]):
    """
    consumer handler for fetch-posts queue
    """
    subreddits = get_subreddits_from_file()
    delay = 70
    for subreddit in subreddits:
        try:
            posts = get_reddit_posts(subreddit)
            if posts and len(posts) > 0:
                insert_to_mongodb(posts, "reddit_posts")
            postids = []
            for post in posts:
                postids.append(post.get("_id"))
            produce_faktory_job("handle_fetch_comments", "fetch-comments", delay, postids)
            delay = delay + 200                            
        except Exception as e:
            logger.error(f"Failed to fetch posts for {subreddit}: {e}")
    produce_faktory_job("handle_fetch_posts", "fetch-posts", 900, [])


def handle_fetch_comments(*pids):
    """
    consumer handler for fetch-comments queue
    """
    if pids and len(pids)>0:
        for postid in pids:
            try:
                id = postid.split('_')[1]
                comments = fetch_and_process_comments(id)
                if comments and len(comments) > 0:
                    insert_to_mongodb(comments, "reddit_comments")
            except Exception as e:
                logger.error(f"Handler failed to fetch comments for {postid}: {e}")


def produce_faktory_job(job_type, queue, schedule_in_seconds, args=[]):
    """
    Produce a job to Faktory queue
    """

    try:
        with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
            producer = Producer(client=client)           
            run_at = datetime.now(timezone.utc) + timedelta(seconds=schedule_in_seconds)
            run_at = run_at.isoformat()           
            job = Job(
                jobtype=job_type,
                args=args,
                queue=queue,
                at=run_at
            )
            producer.push(job)
            logger.info(f"Job '{job_type}' scheduled successfully on queue '{queue}' at {run_at}.")
            return f"success"
    except Exception as e:
        logger.error(f"Failed to produce job: {e}")
        return f"failure"

#Consumers
if __name__ == "__main__":
    # we want to pull jobs off the queues and execute them
    # FOREVER (continuously)
    # handle_fetch_posts(arg=[])
    with Client(faktory_url=FAKTORY_SERVER_URL, role="consumer") as client:
        consumer = Consumer(client=client, queues=["fetch-posts", "fetch-comments"], concurrency=2)
        consumer.register("handle_fetch_posts", handle_fetch_posts)
        consumer.register("handle_fetch_comments", handle_fetch_comments)
        consumer.run()
    
