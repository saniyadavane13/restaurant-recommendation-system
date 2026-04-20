from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"D:\NewDatasetforResturant\zomato.csv")

# Keep columns
df = df[['name', 'cuisines', 'rate', 'approx_cost(for two people)']]

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(subset='name', inplace=True)

df['name'] = df['name'].astype(str)
df['cuisines'] = df['cuisines'].astype(str)

# TF-IDF for cuisines
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['cuisines'])

similarity = cosine_similarity(matrix)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['restaurant'].strip().lower()

    results = []

    # Search by restaurant name
    name_match = df[df['name'].str.lower() == query]

    if not name_match.empty:
        idx = name_match.index[0]
        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]

        for i in scores:
            row = df.iloc[i[0]]
            results.append({
                'name': row['name'],
                'cuisines': row['cuisines'],
                'rating': row['rate'],
                'cost': row['approx_cost(for two people)']
            })

        return render_template("results.html", results=results, msg="Recommended Similar Restaurants")

    # Search by cuisine type
    cuisine_match = df[df['cuisines'].str.lower().str.contains(query, na=False)].head(10)

    if not cuisine_match.empty:
        for _, row in cuisine_match.iterrows():
            results.append({
                'name': row['name'],
                'cuisines': row['cuisines'],
                'rating': row['rate'],
                'cost': row['approx_cost(for two people)']
            })

        return render_template("results.html", results=results, msg="Top Cuisine Matches")

    return render_template("results.html", results=[], msg="No restaurants found")

if __name__ == "__main__":
    app.run(debug=True)