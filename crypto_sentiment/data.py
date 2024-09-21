import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from newsapi import NewsApiClient
import pendulum
from tqdm import tqdm
import os
from transformers import pipeline
import json
import io
import base64
from datetime import datetime
import requests
from notion_api.notionhelper import NotionHelper
nh = NotionHelper()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# My own modules
from crypto_sentiment.params import *
from crypto_sentiment.utils import *

api = NewsApiClient(os.environ.get('NEWSAPI_API_KEY'))

coins = [
    "bitcoin",
    "ethereum",
    "ripple",
    "dogecoin",
    "cardano",
    "polygon",
    "binance",
    "polkadot",
    "uniswap",
    "litecoin",
    "chainlink",
    "solana",
    "stellar",
]

def date_yesterday():
# Get today's date and subtract one day
    date_yesterday = pendulum.now().subtract(days=1).format('YYYY-MM-DD')
    return date_yesterday

@time_it
def get_news(coins):
    master_json_list = []
    for coin in coins:
        print(f"🅾️ {coin} - getting news...")
        output_json = api.get_everything(q=coin, language="en", from_param=date_yesterday, sort_by='relevancy')
        master_json_list.append(output_json)

    return master_json_list

@time_it
def build_news_df(master_json_list, date_yesterday):
    # Initialize the lists to store data
    list_date = []
    list_coin = []
    list_total_results = []
    list_published_at = []
    list_source = []
    list_title = []
    list_content = []
    list_url = []

    # Iterate through the master JSON list and coins
    for index, output_json in enumerate(master_json_list):

        # Skip this iteration if there's a JSON error
        total_results = output_json.get('totalResults', 0)  # Safely get total results
        coin = coins[index]  # Get corresponding coin

        # Check if there are any articles
        if total_results > 0:
            # Iterate through the articles
            articles = output_json.get('articles', [])  # Safely get articles list
            for article in articles:
                # Safely extract values using .get()
                source = article.get('source', {}).get('name', 'Unknown Source')
                title = article.get('title', 'No Title')
                published_at = article.get('publishedAt', 'Unknown Date')
                content = article.get('content', 'No Content')
                url = article.get('url', 'No URL')

                # Append the values to the respective lists
                list_date.append(date_yesterday)
                list_coin.append(coin)
                list_total_results.append(total_results)
                list_published_at.append(published_at)
                list_source.append(source)
                list_title.append(title)
                list_content.append(content)
                list_url.append(url)
        else:
            # If no articles, append default values for the missing data
            list_date.append(date_yesterday)
            list_coin.append(coin)
            list_total_results.append(0)
            list_published_at.append('')
            list_source.append('')
            list_title.append('')
            list_content.append('')
            list_url.append('')

    # Create the DataFrame using the collected data
    data = {
        'date': list_date,
        'coin': list_coin,
        'total_results': list_total_results,
        'published_at': list_published_at,
        'source': list_source,
        'title': list_title,
        'content': list_content,
        'url': list_url
    }

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


@time_it
def sentiment_score_with_transformers(df):
    pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    label = []
    score = []
    true_score = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Sentiment Analysis', smoothing=0.1):
        title = row['title']
        news = title
        if title != '':
            output = pipe(news)

            label.append(output[0]['label'])
            score.append(output[0]['score'])
            if output[0]['label'] == 'neutral':
                true_score.append(0.0)
            elif output[0]['label'] == 'negative':
                true_score.append(-output[0]['score'])
            else:
                true_score.append(output[0]['score'])
        else:
            label.append('neutral')
            score.append(0.0)
            true_score.append(0.0)

    df['label'] = label
    df['score'] = score
    df['true_score'] = true_score
    return df

@time_it
def sentiment_by_coin_df(df, date_yesterday):
    # Function to calculate sentiment score
    def calculate_sentiment_score(P, N, Ne):
        if Ne >= 1.0:
            raise ValueError("Neutral score must be less than 1.0")

        denominator = 1.0 - Ne
        adjusted_positive = P / denominator
        adjusted_negative = N / denominator
        sentiment_score = (adjusted_positive - adjusted_negative) * denominator
        return sentiment_score

    # Function to apply to each group
    def apply_sentiment_calculation(group):
        overall_sentiment = 0
        for _, row in group.iterrows():
            P = row["score"] if row["label"] == "positive" else 0
            N = row["score"] if row["label"] == "negative" else 0
            Ne = row["score"] if row["label"] == "neutral" else 0

            try:
                sentiment = calculate_sentiment_score(P, N, Ne)
            except ValueError:
                sentiment = 0  # If Neutral is 1, consider it neutral
            overall_sentiment += sentiment

        # Return the average sentiment score for the group
        return overall_sentiment / len(group)

    # Group by the entity (e.g., Bitcoin, Ethereum, etc.) and calculate sentiment
    by_coin = df.groupby(['coin', 'total_results']).apply(apply_sentiment_calculation).reset_index(name='overall_sentiment_score')
    total_articles = by_coin['total_results'].sum()
    print(total_articles)
    by_coin['date'] = date_yesterday
    by_coin['weighted_score'] = (by_coin['overall_sentiment_score'] * by_coin['total_results']) / total_articles * 20
    print(by_coin)
    return by_coin


@time_it
def read_txt_file(file_path = 'data/jsonformatter.json'):
    # Define the file path (assuming the file is in the current directory or mounted location)

    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

# Display the structure of the JSON file (first level)
    return data


@time_it
def save_sns_plot(df_grouped):

    mean_sentiment = df_grouped['weighted_score'].mean()
    filename = f"sentiment_plot.png"
    file_path = f"data/{filename}"
    # Create the bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df_grouped,
        y='coin',  # Make sure your column is named 'coin'
        x='weighted_score',
        color="#4d81a6",
        dodge=True# Bar color
    )

    # Draw a vertical red line at the mean sentiment score
    plt.axvline(mean_sentiment, color='#bb271a', linewidth=1)

    # # Add text "MEAN SENTIMENT" on the red
    # plt.text(
    #     mean_sentiment,            # X position (mean sentiment line)
    #     plt.gca().get_ylim()[1] * 1.05,  # Y position (slightly below the top of the plot)
    #     'MEAN SENTIMENT',          # The text to display
    #     color='#000000',               # Text color
    #     rotation=0,               # Rotate the text vertically to match the line
    #     verticalalignment='bottom', # Align vertically centered on the line
    #     fontsize=6
    # )
    plt.tight_layout()
    # Generate a unique file name based on the current date and time

    # Save the plot to a file
    plt.savefig(file_path)

    return file_path, filename

@time_it
def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@time_it
def upload_image_to_github(token, repo, message, content, branch="master", timestamp=timestamp):

    path = f"images/sentiment_plot_{timestamp}.png"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {
        "message": message,
        "content": content,  # The Base64 encoded image
        "branch": branch
    }

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code == 201:
        print("Image uploaded successfully.")
    else:
        print(f"Failed to upload image: {response.json()}")

    return path

@time_it
def create_notion_db_page(df, date_yesterday, date_sentiment_mean, date_sentiment_sum):
    properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": f"Day: {date_yesterday}"
                        }
                    }
                ]
            },
            "Date": {
                "date": {
                    "start": date_yesterday
                }
            },
            "bitcoin": {
                "number": round(df[df['coin'] == 'bitcoin']['weighted_score'].iloc[0], 3)
            },
            "binance": {
                "number": round(df[df['coin'] == 'binance']['weighted_score'].iloc[0], 3)
            },
            "cardano": {
                "number": round(df[df['coin'] == 'cardano']['weighted_score'].iloc[0], 3)
            },
            "chainlink": {
                "number": round(df[df['coin'] == 'chainlink']['weighted_score'].iloc[0], 3)
            },
            "dogecoin": {
                "number": round(df[df['coin'] == 'dogecoin']['weighted_score'].iloc[0], 3)
            },
            "ethereum": {
                "number": round(df[df['coin'] == 'ethereum']['weighted_score'].iloc[0], 3)
            },
            "litecoin": {
                "number": round(df[df['coin'] == 'litecoin']['weighted_score'].iloc[0], 3)
            },
            "polkadot": {
                "number": round(df[df['coin'] == 'polkadot']['weighted_score'].iloc[0], 3)
            },
            "ripple": {
                "number": round(df[df['coin'] == 'ripple']['weighted_score'].iloc[0], 3)
            },
            "solana": {
                "number": round(df[df['coin'] == 'solana']['weighted_score'].iloc[0], 3)
            },
            "stellar": {
                "number": round(df[df['coin'] == 'stellar']['weighted_score'].iloc[0], 3)
            },
            "uniswap": {
                "number": round(df[df['coin'] == 'uniswap']['weighted_score'].iloc[0], 3)
            },
            "polygon": {
                "number": round(df[df['coin'] == 'polygon']['weighted_score'].iloc[0], 3)
            },
            "mean()": {
                "number": round(date_sentiment_mean, 3)
            },
            "sum()": {
                "number": round(date_sentiment_sum, 3)
            },
        }

    returned_json = nh.new_page_to_db('3355789d81ce4f48ad0e9a7a73847ae9', page_properties=properties)
    page_id = returned_json['id']
    return page_id

def chunk_blocks(blocks, chunk_size=100):
    for i in range(0, len(blocks), chunk_size):
        yield blocks[i:i + chunk_size]

@time_it
def write_blocks(df, page_id, timestamp):
    blocks = [{
        "object": "block",
        "type": "heading_1",
        "heading_1": {
            "rich_text": [
                {"type": "text", "text": {"content": "Today's Cryptocurrency News"}}
            ]
        },
    },
    {
    "type": "image",
    "image": {
        "type": "external",
        "external": {
            "url": f"https://github.com/janduplessis883/crypto-sentiment/blob/master/images/sentiment_plot_{timestamp}.png?raw=true"
        }
    }
    }]

    for index, row in df.iterrows():
        title = row['title']
        coin = row['coin']
        url = row['url']
        label = row['label']

        if label == 'neutral':
            indicator = '⚪️'
        elif label == 'positive':
            indicator = '🟢'
        else:
            indicator = '🔴'

        block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"{indicator} - {coin.upper()} - {title}",
                                "link": {"url": url} if url else None
                            }
                        }
                    ]
                }
            }

        blocks.append(block)

    for chunk in chunk_blocks(blocks, 100):
        nh.append_page_body(page_id, chunk)
        print(f"✅ Block to Notion")

if __name__ == "__main__":
    testing = False

    date_yesterday = date_yesterday()

    if testing == False:
        master_json_list = get_news(coins)
    elif testing == True:
        print("👷🏻❌❌ TESTING MODE - bypass NEWSAPI")
        master_json_list = read_txt_file()

    df = build_news_df(master_json_list, date_yesterday)

    df = sentiment_score_with_transformers(df)
    print("💾 Sentiment Score DF saved")
    df.to_csv('data/sentiment_score_' + date_yesterday + '.csv', index = False)

    by_coin = sentiment_by_coin_df(df, date_yesterday)
    by_coin.to_csv('data/by_coin.csv', index=False)
    print("💾 by_coin DF saved")

    date_sentiment_sum = by_coin['weighted_score'].sum()
    print(f"➡️ Total Sentiment: {date_sentiment_sum}")
    date_sentiment_mean = by_coin['weighted_score'].mean()
    print(f"Ⓜ️  Mean Sentiment: {date_sentiment_mean}")

    file_path, filename = save_sns_plot(by_coin)
    logger.info(f"📷 Sentiment Score Plot saved to {file_path}")

    # Example usage:
    encoded_image = encode_image_to_base64(file_path)

    token = os.environ.get('GITHUB_TOKEN_CUSTOM_1')
    repo = "janduplessis883/crypto-sentiment"
    message = "Adding a new image"
    branch = "master"
    path = upload_image_to_github(token, repo, message, encoded_image, branch, timestamp)

    page_id = create_notion_db_page(by_coin, date_yesterday, date_sentiment_mean, date_sentiment_sum)
    write_blocks(df, page_id, timestamp)
