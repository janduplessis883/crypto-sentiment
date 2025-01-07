# **Crypto-Sentiment**
![Image](images/logo.png)
## Overview
Crypto-Sentiment uses NEWSAPI to fetch the latest cryptocurrency news and performs sentiment analysis on the gathered articles. The sentiment analysis is weighted based on the number of articles and the results are uploaded to Notion for archiving and plotting. Additionally, individual article links are saved to the Notion page for later reference.
Crypto-Sentiment is a Python package designed to analyze the sentiment of cryptocurrency-related news and social media posts. By leveraging natural language processing (NLP) techniques, this package provides insights into the overall market sentiment, which can be useful for traders and investors.

## Features

- Sentiment analysis of cryptocurrency news articles
- Sentiment analysis of social media posts related to cryptocurrencies
- Aggregated sentiment scores for different cryptocurrencies
- Visualization of sentiment trends over time

## Installation

To install the package, use pip:

```bash
pip install crypto-sentiment
```

## Usage

Here's a basic example of how to use the Crypto-Sentiment package:

```python
from crypto_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
news_sentiment = analyzer.analyze_news("Bitcoin hits a new all-time high!")
social_media_sentiment = analyzer.analyze_social_media("#Bitcoin to the moon!")

print("News Sentiment:", news_sentiment)
print("Social Media Sentiment:", social_media_sentiment)
```

## Reference

For full functionality, refer to the `data.py` file. This file contains all the necessary functions and classes to perform sentiment analysis on cryptocurrency-related data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact the project maintainer at drjanduplessis@icloud.com
