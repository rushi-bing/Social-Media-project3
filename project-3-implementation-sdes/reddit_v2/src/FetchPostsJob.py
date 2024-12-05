

from RedditApiCalls import fetch_subreddit_data
from datetime import datetime, timezone
from logger_setup import setup_logger
from ToxicityApiClient import analyze_toxicity


logger = setup_logger("reddit", log_file='logs/reddit.log', max_bytes=10*1024*1024, backup_count=5)


def extract_post_info(data):
    """Extracts relevant fields from a Reddit post JSON object."""
    post = {
        "author": data.get("author"),
        "awards_count": data.get("total_awards_received", 0),
        "_id": data.get("name"),
        "score": data.get("score", 0),
        "selftext": data.get("selftext", ""),
        "subreddit": data.get("subreddit"),
        "title": data.get("title"),
        "ups": data.get("ups", 0),
        "upvote_ratio": data.get("upvote_ratio", 0.0),
        "utc": datetime.fromtimestamp(data.get("created_utc", 0.0), timezone.utc).isoformat()
    }

    # Add toxicity analysis
    post["toxicity"] = analyze_toxicity(post["title"] + " " + post["selftext"])

    return post


def get_reddit_posts(subreddit, before=None):
    """
    Calls fetch_subreddit_data and extracts an array of posts with the required fields.
    """
    try:
        response = fetch_subreddit_data(subreddit, before)
        posts_data = response.get("data", {}).get("children", [])
        posts = [
            extract_post_info(post.get("data", {}))
            for post in posts_data
            if post.get("kind") == "t3"
        ]
    except Exception as e:
        logger.error(f"Failed to extract posts for subreddit {subreddit}: {e}")
        return []
    return posts
