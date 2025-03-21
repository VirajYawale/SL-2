## SL-2

All About Machine Learning
REFER: https://youtube.com/playlist?list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&si=0-wTWXOgsey9Ht_e

<hr>

### Extra Learning 


<p><b> 1. How to make the dataset/dataframe from the API:</b> (refer: https://youtu.be/roTZJaxjnJc?si=23Z_jSg_6za7O1fV ) </p>

<p>Using the data from Rapid API website. <b>( Use its python code snippets and it not subscribed then subscribe it)</b></p>

![img](image/img1.png)

<b>this is where we come to know where the data is in API:</b>

![img](image/img2.png)

In the data have data in different pages then we have to use the loop

<b>Data: ![img](image/img3.png)</b>

<b>Code: ![img](image/img4.png)  (here, we will change the page number in loop and append the data in dataframe)</b>

<b>We can convert the data to csv file too</b>


```bash
    df.to_csv('top_news.csv')
```

<h4>This dataset can be aploaded on kaggle. (which can boost your kaggle profile)</h4>


Note: 

![img](image/img5.png)


<h4><b>In real_time_fetch.ipnyb: we fetch the data from the real time api (https://newsdata.io/api-key) And in code if there is new news formed we add it to csv file <br> but if there is no new news then we don't add the news repeatedly.</b></h4>

<p>But this can cause the big size csv file, so we should have to use the online data.</p>
<h3><b>To prevent the big csv file:</b>  <br>
</h3>
        1.Limit the Number of Entries in the CSV

```python

    MAX_ROWS = 5000  # Keep only the last 5000 rows

    def save_news_to_csv(news_data, filename=NEWS_CSV):
        if not news_data:
            print("✅ No new news updates.")
            return
        
        df = pd.DataFrame(news_data)
        
        # Append new data
        df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))
        
        # Limit the file size
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            if len(df_existing) > MAX_ROWS:
                df_existing = df_existing.tail(MAX_ROWS)  # Keep only the last 5000 rows
                df_existing.to_csv(filename, index=False)
        
        print(f"✅ {len(news_data)} new articles added. Keeping last {MAX_ROWS} records.")


```


<br>

    2. Use a Database Instead of CSV (Recommended): Instead of saving everything to a CSV file, you can store it in a lightweight database like SQLite.

```python

import sqlite3

DB_NAME = "news.db"

# Initialize database
def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    title TEXT,
                    description TEXT,
                    url TEXT UNIQUE,
                    published_at TEXT,
                    prediction TEXT
                )''')
    conn.commit()
    conn.close()

# Save news to SQLite instead of CSV
def save_news_to_db(news_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    for article in news_data:
        try:
            c.execute("INSERT INTO news (source, title, description, url, published_at, prediction) VALUES (?, ?, ?, ?, ?, ?)",
                      (article["source"], article["title"], article["description"], article["url"], article["published_at"], article["prediction"]))
        except sqlite3.IntegrityError:
            pass  # Avoid duplicate entries
    
    conn.commit()
    conn.close()
    print(f"✅ {len(news_data)} new articles added to the database.")

# Initialize DB before starting
initialize_db()

```

<br>

    ```3. Auto-Delete Old Data After a Certain Period:```



```python
    import datetime

def delete_old_news(days=7):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Calculate the date threshold
    threshold_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Delete old entries
    c.execute("DELETE FROM news WHERE published_at < ?", (threshold_date,))
    conn.commit()
    conn.close()
    print(f"🗑️ Deleted news older than {days} days.")

# Call this function periodically
delete_old_news()

```


<b> Here we will use 2nd method: we will store it in database. </b>