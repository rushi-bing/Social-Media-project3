import base64
import requests
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("reddit", log_file='logs/reddit.log', max_bytes=10*1024*1024, backup_count=5)

CLIENT_ID = "Z1hwVQGAVBpf9wAS4Ylqfg"
CLIENT_SECRET = "X-aK_Ay-HKF8QD1dUmmbeNhybH415g"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE_URL = "https://oauth.reddit.com"
USER_AGENT = "springboot-app:com.JobMarketETL.redditfetcher:v1.0 (by /u/Terrible-College-665)"

def get_reddit_access_token():
    """
    Fetches the Reddit API access token using client credentials.
    """
    auth = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT
    }
    payload = {
        "grant_type": "client_credentials"
    }
    response = requests.post(TOKEN_URL, headers=headers, data=payload)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        return access_token
    else:
        raise Exception(f"Failed to fetch access token: {response.status_code} - {response.text}")
    

def fetch_subreddit_data(subreddit, before=None):
    """
    Fetches posts from the given subreddit using the access token.
    The JSON response from Reddit's API containing posts.
    """
    access_token = get_reddit_access_token()
    url = f"{REDDIT_API_BASE_URL}/r/{subreddit}/new?limit=50&raw_json=1"
    if before:
        url += f"&before={before}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error while fetching subreddit data: {response.status_code} - {response.text}")
            raise Exception("Failed to fetch subreddit data.")

    except Exception as e:
        print(f"Error while fetching subreddit data: {e}")
        return {}


def fetch_comments(post_id):
    """Fetches comments for the supplied post ID from Reddit."""
    access_token = get_reddit_access_token()
    url = f"{REDDIT_API_BASE_URL}/comments/{post_id}?raw_json=1"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            json_array = response.json()
            # Check if we have more than one element as first element is for post itself
            if len(json_array) > 1:
                comments_data = json_array[1].get("data", {}).get("children", [])
                comments = [
                    comment.get("data", {})
                    for comment in comments_data
                ]
                return comments
            else:
                return []
        else:
            logger.error(f"Error while fetching comments: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Exception occurred while fetching comments: {e}")
        return []
