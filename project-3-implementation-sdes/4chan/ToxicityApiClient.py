import requests
from logger_setup import setup_logger

# Setup logger
logger = setup_logger("ToxicityAnalysis", log_file='logs/toxicity_analysis.log', max_bytes=10*1024*1024, backup_count=5)

API_URL = "https://api.moderatehatespeech.com/api/v1/moderate/"
API_KEY = "cd648eb2f85cfcff2259ab17d97ce144"  # Replace with your API key or use env variables as needed


def analyze_toxicity(text, max_retries=3):
    """
    Analyze the toxicity of a given text using the ModerateHatespeech API.

    Parameters:
        text (str): The text content to analyze.
        max_retries (int): Maximum retry attempts for timeouts or transient errors.

    Returns:
        dict: A dictionary containing the toxicity analysis result, or default_value if analysis fails.

    Example Response:
        {
            "class": "flag",
            "confidence": 0.87
        }
    """
    default_value={"class": "neutral", "confidence": 0.0}
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("Invalid or empty text provided for toxicity analysis.")
        return default_value

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "token": API_KEY,
        "text": text
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=10)

            # Check for HTTP errors
            response.raise_for_status()

            # Parse the JSON response
            try:
                result = response.json()
            except ValueError:
                print(f"Error: Invalid JSON response: {response.text}")
                return default_value

            # Validate expected keys in response
            if "class" in result and "confidence" in result:
                # logger.info(f"Toxicity analysis successful: {result}")
                return {
                    "class": result["class"],
                    "confidence": result["confidence"],
                }
            elif "error" in result:
                logger.error(f"API returned an error: {result.get('error')}")
                return default_value
            else:
                logger.error(f"Unexpected response format: {result}")
                return default_value

        except requests.exceptions.Timeout:
            logger.warning(f"Toxicity analysis request timed out. Retry {retries + 1} of {max_retries}.")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error during toxicity analysis: {conn_err}")
            break  # Connection issues may not benefit from retries
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            break  # Stop retries for HTTP errors
        except requests.exceptions.RequestException as req_err:
            logger.error(f"General request exception occurred: {req_err}")
            break  # Stop retries for unknown request exceptions
        except ValueError as json_err:
            logger.error(f"Error parsing JSON response: {json_err}")
            break  # Likely a malformed response, so stop retries
        except Exception as e:
            logger.error(f"Unexpected error during toxicity analysis: {e}")
            break  # Stop retries for unexpected errors

        retries += 1

    logger.error("Toxicity analysis failed after maximum retries.")
    return default_value
