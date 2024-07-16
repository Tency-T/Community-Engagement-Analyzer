import pandas as pd
import numpy as np
import seaborn as sns
import calendar
import emoji
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def fetch_stats(selected_user, df):
    if selected_user == "Overall":
        num_messages = df.shape[0]
        num_media = df[df['is_media']].shape[0]
        num_links = df[df['is_link']].shape[0]
    else:
        num_messages = df[df['user'] == selected_user].shape[0]
        num_media = df[(df['user'] == selected_user) & (df['is_media'])].shape[0]
        num_links = df[(df['user'] == selected_user) & (df['is_link'])].shape[0]

    return num_messages, num_media, num_links

def calculate_average_response_time(selected_user, df):
    filtered_df = df if selected_user == "Overall" else df[df['user'] == selected_user]
    
    if filtered_df.shape[0] < 2:
        return "N/A"
    
    response_times = filtered_df['date'].diff().mean()
    
    # Convert timedelta to hours, minutes, seconds
    average_response_hours = int(response_times.seconds // 3600)
    average_response_minutes = int((response_times.seconds % 3600) // 60)
    average_response_seconds = int(response_times.seconds % 60)
    
    # Format into a human-readable string
    average_response_time_str = "{:02}:{:02}:{:02}".format(average_response_hours, average_response_minutes, average_response_seconds)
    
    return average_response_time_str


def find_most_active_contact(selected_user, df):
    if df.empty or df['user'].empty:
        return "No messages found"

    if selected_user == "Overall":
        if df['user'].empty:
            return "No messages found"
        else:
            return df['user'].value_counts().idxmax()
    else:
        return selected_user  # If selected_user is specific, return it as the most active contact


# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(message):
    """
    Analyzes sentiment of a message and returns the sentiment label.
    """
    result = sentiment_pipeline(message)
    return result[0]['label']

def most_common_sentiment(df, selected_user="Overall"):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    
    positive_count = 0
    neutral_count = 0
    negative_count = 0  

    for message in df['message']:
        sentiment = analyze_sentiment(message)
        if sentiment in ['5 stars', '4 stars']:
            positive_count += 1
        elif sentiment == '3 stars':
            neutral_count += 1
        else:
            negative_count += 1
    
    if positive_count > neutral_count and positive_count > negative_count:
        return "Positive"
    elif neutral_count > positive_count and neutral_count > negative_count:
        return "Neutral"
    else:
        return "Negative"
    

#def monthly_timeline(df):
    #monthly_counts = df.resample('M', on='date')['message'].count()
    # Ensure all 12 months are present, even if some have zero messages
    #monthly_counts = monthly_counts.reindex(pd.date_range(start=monthly_counts.index.min(), end=monthly_counts.index.max(), freq='M'), fill_value=0)
    #return monthly_counts

def monthly_timeline(df):
    # Group by month and count messages
    monthly_counts = df.resample('M', on='date')['message'].count()
    
    # Ensure all months are present, even if some have zero messages
    min_date = df['date'].min().replace(day=1)
    max_date = df['date'].max().replace(day=1)
    # Reindex to fill missing months with zero messages
    monthly_counts = monthly_counts.reindex(pd.date_range(start=monthly_counts.index.min(), end=monthly_counts.index.max(), freq='M'), fill_value=0)
    
    return monthly_counts


def activity_map(df):
    # Most busy day
    most_busy_day = df['date'].dt.day_name().value_counts().sort_values(ascending=False)
    
    # Most busy month
    most_busy_month = df['date'].dt.month_name().value_counts().sort_index()
    
    return most_busy_day, most_busy_month

def weekly_activity_map(df):
    # Ensure all days of the week are present, even if some have zero messages
    heatmap_data = df.groupby([df['date'].dt.dayofweek, df['date'].dt.hour])['message'].count().unstack().fillna(0)
    heatmap_data.index = [calendar.day_name[i] for i in range(7)]  # Convert day of week to names
    
    # Sort columns by hour of day (ascending)
    heatmap_data = heatmap_data.reindex(columns=range(24))
    
    return heatmap_data

def most_busy_users(df):
    user_counts = df['user'].value_counts().head(5)
    total_messages = user_counts.sum()
    user_percentage = (user_counts / total_messages) * 100
    return user_counts, user_percentage

def generate_wordcloud(df):
    text = ' '.join(df['message'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def plot_activity_heatmap(heatmap_data):
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Number of Messages'})
    plt.title('Weekly Activity Map')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    return plt

def plot_most_busy_users(user_counts):
    plt.figure(figsize=(10, 6))
    user_counts.plot(kind='bar', rot=0)
    plt.xlabel('User')
    plt.ylabel('Number of Messages')
    plt.title('Top 5 Busy Users')
    plt.tight_layout()
    return plt

def activity_map(df):
    # Most busy day
    most_busy_day = df['date'].dt.date.value_counts().idxmax()
    
    # Most busy month
    most_busy_month = df['date'].dt.to_period('M').value_counts().idxmax()
    
    return most_busy_day, most_busy_month

def plot_busiest_day(df):
    daily_counts = df['date'].dt.day_name().value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    daily_counts.plot(kind='bar', rot=0)
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Messages')
    plt.title('Messages per Day of Week')
    plt.tight_layout()
    return plt


def plot_busiest_month(df, most_busy_month):
    monthly_counts = df.resample('D', on='date')['message'].count()
    
    plt.figure(figsize=(10, 6))
    monthly_counts.plot(kind='line', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.title(f'Messages per Day in {most_busy_month}')
    plt.tight_layout()
    return plt

def get_emoji(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df