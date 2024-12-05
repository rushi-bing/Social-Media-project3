## 4Chan Crawler

# Steps to Run Crawler:

1. navigate to /4chan folder and create dev environment to keep everything clean:

   ```bash
   python -m venv ./env/dev
   ```

2. Activate new dev environment:

   ```bash
   source env/dev/bin/activate
   ```

   to Deactivate dev environment

   ```bash
   deactivate
   ```

3. Install requirements.txt all dependcies:

   ```bash
   pip install -r requirements.txt
   ```

4. Maker sure .env file is setup it all follwoing parameters

   - FAKTORY_SERVER_URL - faktory localhost url
   - MONGODB_URI - mongoDB connection url
   - MONGODB_DATABASE_NAME - mongoDB database name
   - MONGODB_POST_COLLECTION_NAME - mongoDB collection name to store

5. Start docker faktory and add faktory image

   ```bash
   docker run -it --name faktory \
   -v ~/projects/docker-disks/faktory-data:/var/lib/faktory/db \
   -e "FAKTORY_PASSWORD=password" \
   -p 127.0.0.1:7419:7419 \
   -p 127.0.0.1:7420:7420 \
   contribsys/faktory:latest \
   /faktory -b :7419 -w :7420
   ```

6. Test Faktory setup

   ```bash
   python3 faktory-test.py `

   ```

7. Start 4chan crawler

   ```bash
   python3 chan_crawler.py

   ```

8. Cold Start the 4chan board crawler

   ```bash
   python3 cold_start_board.py board_name
   ```

   g - is Technology board

   add board_name on which crawler should work on

   Refer: [4Chan Boards](https://4chanarchives.com/boards)


# Step-by-Step Guide to Using screen for Running a Python Script:
used for: Python script continues running even if you disconnect from your SSH session or close your terminal.

1. Install screen
   ```bash
      sudo apt-get install screen
   ```

2. Start a new screen session
   ```bash
    screen -S job_post_crawler
   ```
3. Run your Python script in screen -
   ```bash
   python3 job_post_crawler.py
   ```
4. Detach from the screen session by
   ```bash
   Ctrl + A, then D
   ```
5. Reattach to your screen session (optional)
   ```bash
   screen -r my_data_collection
   ```
6. List active screen sessions
   ```bash
    screen -ls
    ```
7. screen command to kill the session
    ```bash
    screen -S job_post_crawler -X quit
    ```

References:
https://copeid.ssrc.msstate.edu/wp-content/uploads/2022/06/FINAL-4chan-Documentation.pdf
https://github.com/4chan/4chan-API/blob/master/pages/Threads.md
