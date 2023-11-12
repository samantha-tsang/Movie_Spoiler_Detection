import pandas as pd
import json
import numpy as np
import csv
from sklearn.model_selection import train_test_split

##### READ IN DATA
# data frame with movie details: plot summary, duration, genre, rating, release date, and plot synopsis
movie_details_json = []

with open('IMDB_movie_details.json', 'r') as file:
    for line in file:
        try:
            json_object = json.loads(line)
            movie_details_json.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Convert the list of dictionaries to a DataFrame
details_df = pd.DataFrame(movie_details_json)

# data frame with movie reviews detail: review data, review text, rating, and review summary
# contains target variable "is_spoiler"
movie_reviews_json = []
with open('IMDB_reviews.json', 'r') as file:
    for line in file:
        try:
            json_object = json.loads(line)
            movie_reviews_json.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Convert the list of dictionaries to a DataFrame
reviews_df = pd.DataFrame(movie_reviews_json)

# delete json items to save space
del json_object
del movie_details_json
del movie_reviews_json

##### merge ntwo df on movie id
df = pd.merge(reviews_df, details_df, on  = "movie_id")

##### Brief EDA

# number of movies
len(set(df['movie_id'])) # there are 1570 unique movies in data

# number of spoilers
sum(df["is_spoiler"].apply(int)) # about 15K spoilerds, roughly 26% of the data

# is synopsis or summary more details?
np.mean(df['plot_summary'].apply(len))
np.mean(df['plot_synopsis'].apply(len)) # synopsis is way longer than summary, use synopsis
df = df.drop(columns = 'plot_summary')

# combine the synopsis and review text using plot: (synopsis) review:(review)
def combine_plot_review(row):
    synop = row['plot_synopsis']
    review = row['review_text']
    return "plot " + synop + "review " + review
df['combined_text'] = df.apply(combine_plot_review, axis = 1)

##### split train test
X = df.loc[:, df.columns != "is_spoiler"]
y = df["is_spoiler"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 123)


X_train.to_json("X_train.json", orient='records', lines=True)
X_test.to_json("X_test.json", orient='records', lines=True)
with open("y_train.json", "w") as f:
    json.dump(f, y_train)
with open("y_test.json", "w") as f:
    json.dump(f, y_test)
