from RedditApiCalls import fetch_comments
from logger_setup import setup_logger
from datetime import datetime, timezone
from ToxicityApiClient import analyze_toxicity

# Setup logger
logger = setup_logger("reddit", log_file='logs/reddit.log', max_bytes=10*1024*1024, backup_count=5)

comments_list = []
def generate_comment_list(comment):
    """Recursively generates a list of relevant comments from the comment JSON."""
    # Check if the comment has a body
    if 'body' in comment:
        reddit_post_comment = {}
        try:
            reddit_post_comment = {
                "_id": comment.get("name"),
                "subreddit": comment.get("subreddit"),
                "post_id": comment.get("link_id"),
                "author": comment.get("author"),
                "body": comment.get("body"),
                "score": comment.get("score"),
                "parent_id": comment.get("parent_id"),
                "utc": datetime.fromtimestamp(comment.get("created_utc", 0.0), timezone.utc).isoformat()
            }
            # Add toxicity analysis
            reddit_post_comment["toxicity"] = analyze_toxicity(reddit_post_comment["body"])
        except Exception as e:
            logger.error(f"Failed to extract comments for {comment}: {e}")
        comments_list.append(reddit_post_comment)
        # Check if there are replies to the comment
        replies = comment.get("replies")
        if isinstance(replies, dict):  # Ensure replies is a dictionary
            replies_data = replies.get("data", {}).get("children", [])
            for reply in replies_data:
                generate_comment_list(reply.get("data", {}))


def fetch_and_process_comments(post_id):
    """Fetch comments for the given post ID and process them into a list."""
    try:
        comments = fetch_comments(post_id)
        for comment in comments:
            generate_comment_list(comment)
    except Exception as e:
        logger.error(f"Failed to fetch comments for {postid}: {e}")
        return []
    return comments_list