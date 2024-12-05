import requests
import json
import time
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("4chan client", log_file='logs/4chan_crawler.log', max_bytes=10*1024*1024, backup_count=5)

class ChanClient:
    API_BASE = "http://a.4cdn.org"
    last_request_time = 0  # Last request time
    request_interval = 1.0  # API Rate limit 1 second

    def rate_limit(self):
        """
        Enforce rate-limiting by ensuring at least 1 second between API requests.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def get_thread(self, board, thread_number, if_modified_since=None):
        """
        Retrieve JSON data for a specific thread on a board.

        Parameters:
            board (str): The board name (e.g., 'g', 'pol').
            thread_number (int): The thread number.
            if_modified_since (str): The 'If-Modified-Since' timestamp to avoid re-fetching unchanged data.

        Returns:
            dict: JSON data of the thread if successful, None otherwise.
        """
        request_pieces = [board, "thread", f"{thread_number}.json"]
        api_call = self.build_request(request_pieces)

        headers = {}
        if if_modified_since:
            headers['If-Modified-Since'] = if_modified_since

        self.rate_limit()  # rate-limiting
        return self.execute_request(api_call, headers=headers)

    def get_catalog(self, board, if_modified_since=None):
        """
        Retrieve catalog JSON data for a specific board.

        Parameters:
            board (str): The board name.
            if_modified_since (str): The 'If-Modified-Since' timestamp to avoid re-fetching unchanged data.

        Returns:
            dict: JSON data of the catalog if successful, None otherwise.
        """
        request_pieces = [board, "catalog.json"]
        api_call = self.build_request(request_pieces)

        headers = {}
        if if_modified_since:
            headers['If-Modified-Since'] = if_modified_since

        self.rate_limit()  # rate-limiting
        return self.execute_request(api_call, headers=headers)

    def build_request(self, request_pieces):
        """
        Build the API request URL from the provided pieces.
        """
        api_call = "/".join([self.API_BASE] + request_pieces)
        logger.debug(f"Built API call URL: {api_call}")
        return api_call

    def execute_request(self, api_call, headers=None):
        """
        Execute an HTTP GET request and return JSON data.
        
        Returns:
            dict: The JSON response if successful, None otherwise.
        """
        try:
            resp = requests.get(api_call, headers=headers, timeout=10)
            resp.raise_for_status()
            logger.info(f"Request successful: {api_call} - Status code: {resp.status_code}")
            try:
                if resp.status_code == 304:  # Not modified
                    logger.info(f"No new data for {api_call} (304 Not Modified).")
                    return None

                data = resp.json()
                return data
            except json.JSONDecodeError as e:
                logger.exception(f"JSON decoding failed for {api_call}: {e}")
                return None
        except (HTTPError, ConnectionError, Timeout) as e:
            logger.error(f"Request failed for {api_call}: {e}")
            return None
        except RequestException as e:
            logger.exception(f"An error occurred during the request for {api_call}: {e}")
            return None


if __name__ == "__main__":
    client = ChanClient()
    thread_json = client.get_thread("g", 103021049)
    if thread_json:
        print(thread_json)
    else:
        print("Failed to retrieve thread data.")
