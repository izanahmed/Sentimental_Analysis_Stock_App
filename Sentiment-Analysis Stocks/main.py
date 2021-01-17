## Author: Izan Ahmed
## Sentimental Analysis on Stocks of choice

# Libraries
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Class
class StockAnalyser:
    # Initializing the relevant data structures with url
    def __init__(self):
        self.stock_url = 'https://finviz.com/quote.ashx?t='
        self.stocks = []
        self.news_tables = {}
        self.cleaned_data = []

    # Adding a stock to analyse
    def add_stock(self):
        while True:
            s = input("Insert Ticker or press # to continue: ")
            if s == '#':
                break
            else:
                self.stocks.append(s)
        return

    # Getting the information for the stocks from the Url and storing it
    def process_url(self):
        for stock in self.stocks:
            final_url = self.stock_url + stock
            finviz_request = Request(url=final_url, headers={'user-agent': 'my-stock-app'})
            finviz_response = urlopen(finviz_request)
            webpage = BeautifulSoup(finviz_response, features="lxml")
            news = webpage.find(id='news-table')
            self.news_tables[stock] = news

    # Cleaning the data to get date, time and heading
    def cleaning_data(self):
        for stock, news in self.news_tables.items():
            for elm in news.findAll('tr'):
                heading = elm.a.get_text()
                time = elm.td.text.split(' ')
                if len(time) == 2:
                    article_time = time[1]
                    date = time[0]
                else:
                    article_time = time[0]
                self.cleaned_data.append([stock, date, article_time, heading])

    # Function that does the sentimental analysis
    def sentiment(self, text):
        return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

    # Applying previous function to a column in a data frame
    def applying_polarity(self):
        df = pd.DataFrame(self.cleaned_data, columns=['Stock', 'Date', 'Time', 'Heading'])
        df['Polarity Score'] = df['Heading'].apply(self.sentiment)
        df['Date'] = pd.to_datetime(df.Date).dt.date
        return df

    # Plotting the Average polarity Scores for the Stocks per day
    def plotting(self):
        df = self.applying_polarity()
        new_df = df.groupby(['Stock', 'Date']).mean()
        new_df = new_df.unstack()
        final_df = new_df.xs('Polarity Score', axis="columns").transpose()
        final_df.plot(kind='bar')
        plt.show()

    # Running the app
    def run_app(self):
        self.add_stock()
        self.process_url()
        self.cleaning_data()
        self.applying_polarity()
        self.plotting()

# Main
def main():
    Analysis = StockAnalyser()
    Analysis.run_app()

# Running main
if __name__ == '__main__':
    main()

