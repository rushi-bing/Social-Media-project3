
# 4Chan and Reddit Crawler and Analysis

This repository contains tools for crawling and analyzing data from 4Chan and Reddit. Below is the step-by-step guide to set up, run crawlers, and perform analysis.

---

## **4Chan Crawler**

### **Setup**
1. **Create a `.env` file** in the `4chan/` folder to set up parameters:
   ```
   FAKTORY_SERVER_URL=localhost_url
   MONGODB_URI=mongodb_connection_url
   MONGODB_DATABASE_NAME=jobMarketDB
   MONGODB_DB_COLLECTION_NAME=4chan_posts_comments
   POL_COLLECTION_NAME=4chan_politics_comments
   ```

2. Navigate to the `4chan` folder and create a virtual environment:
   ```bash
   python3 -m venv ./env/dev
   ```

3. **Activate the virtual environment:**
   ```bash
   source env/dev/bin/activate
   ```
   **To deactivate the virtual environment:**
   ```bash
   deactivate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Docker Faktory and add Faktory image:**
   ```bash
   docker run -it --name faktory \
   -v ~/projects/docker-disks/faktory-data:/var/lib/faktory/db \
   -e "FAKTORY_PASSWORD=password" \
   -p 127.0.0.1:7419:7419 \
   -p 127.0.0.1:7420:7420 \
   contribsys/faktory:latest \
   /faktory -b :7419 -w :7420
   ```

### **Run Crawler**
1. Start the 4Chan crawler:
   ```bash
   python3 chan_crawler.py
   ```

2. Perform a cold start of the 4Chan board crawler:
   ```bash
   python3 cold_start_board.py board_name
   ```
   Example: `board_name` can be `/g/` (Technology board) or `/pol/` (Politics board).

   **Reference:** [4Chan Boards](https://4chanarchives.com/boards)

---

## **Reddit Crawler**

### **Setup**
1. Navigate to the `reddit_v2` folder and create a virtual environment:
   ```bash
   python3 -m venv ./env/dev
   ```

2. **Activate the virtual environment:**
   ```bash
   source env/dev/bin/activate
   ```
   **To deactivate the virtual environment:**
   ```bash
   deactivate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Docker is installed and running.

### **Run Crawler**
1. Start the Reddit crawler:
   ```bash
   python3 src/FaktoryService.py
   ```
   **Note:** For the initial run, uncomment line #94 in `FaktoryService.py` to add initialization jobs to the Faktory queue.

---

## **Analysis**

### **4Chan Analysis**
1. Navigate to the `4chan` folder.
2. Ensure an `img` folder exists for saving generated images.
3. Activate the virtual environment and install dependencies (as shown in the setup steps).

**Run the following scripts for analysis:**

1. **Sentiment Analysis graph:**
   ```bash
   python3 sentiment_analysis_bar_chart.py
   ```

2. **Toxicity Class Analysis graph:**
   ```bash
   python3 toxicity_class_analysis.py
   ```

3. **Technology board data count graph (`/g` board):**
   ```bash
   python3 tech_board_comment.py
   ```

4. **Hourly comments graph (`/pol` board):**
   ```bash
   python3 pol_1_to_14_hourly_comment.py
   ```

---

### **Reddit Analysis**
1. Navigate to the `reddit_v2` folder.
2. Ensure an `img` folder exists for saving generated images.
3. Activate the virtual environment and install dependencies (as shown in the setup steps).

**Run the following scripts for analysis:**

1. **Sentiment Analysis graph:**
   ```bash
   python3 sentiment_analysis_bar_chart.py
   ```

2. **Toxicity Class Analysis graph:**
   ```bash
   python3 toxicity_class_analysis.py
   ```

3. **Subreddit data distribution graph:**
   ```bash
   python3 subreddit_data_analysis_horizontal_bar.py
   ```

4. **Daily submissions graph (`r/pol` board):**
   ```bash
   python3 pol_1_to_14_daily_posts.py
   ```

5. **Hourly comments graph (`r/pol` board):**
   ```bash
   python3 pol_1_to_14_hourly_comments.py
   ```

---

## **Using `screen` for Persistent Sessions**

### Purpose
Run Python scripts in a terminal session that persists even if disconnected or closed.

### Commands
1. **Install `screen`:**
   ```bash
   sudo apt-get install screen
   ```

2. **Start a new session:**
   ```bash
   screen -S job_post_crawler
   ```

3. **Run a script within the session:**
   ```bash
   python3 job_post_crawler.py
   ```

4. **Detach from the session:**
   ```bash
   Ctrl + A, then D
   ```

5. **Reattach to a session:**
   ```bash
   screen -r job_post_crawler
   ```

6. **List active sessions:**
   ```bash
   screen -ls
   ```

7. **Kill a session:**
   ```bash
   screen -S job_post_crawler -X quit
   ```

---

## **References**
- [4Chan Documentation](https://copeid.ssrc.msstate.edu/wp-content/uploads/2022/06/FINAL-4chan-Documentation.pdf)
- [4Chan API](https://github.com/4chan/4chan-API/blob/master/pages/Threads.md)

--- 
