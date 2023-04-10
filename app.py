from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import csv
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__, static_folder="static")



def trainPopularityModel():
    books = pd.read_csv("./data/books.csv")
    ratings = pd.read_csv("./data/ratings.csv")
    ratings_with_name = ratings.merge(books, on="ISBN")
    num_rating_df = (
        ratings_with_name.groupby("Book-Title").count()["Book-Rating"].reset_index()
    )
    num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)
    avg_rating_df = (
        ratings_with_name.groupby("Book-Title").mean()["Book-Rating"].reset_index()
    )
    avg_rating_df.rename(columns={"Book-Rating": "avg_rating"}, inplace=True)
    popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
    popular_df = (
        popular_df[popular_df["num_ratings"] >= 250]
        .sort_values("avg_rating", ascending=False)
        .head(20)
    )
    popular_df = popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Image-URL-M",
            "num_ratings",
            "avg_rating",
        ]
    ]

    return popular_df


def trainCollaborativeFiltering():
    users = pd.read_csv("./data/users.csv")
    books = pd.read_csv("./data/books.csv")
    ratings = pd.read_csv("./data/ratings.csv")
    ratings_with_name = ratings.merge(books, on="ISBN")
    x = ratings_with_name.groupby("User-ID").count()["Book-Rating"] > 200
    experiencedUsers = x[x].index
    filtered_rating = ratings_with_name[
        ratings_with_name["User-ID"].isin(experiencedUsers)
    ]
    y = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(famous_books)]
    pt = final_ratings.pivot_table(
        index="Book-Title", columns="User-ID", values="Book-Rating"
    )
    pt.fillna(0, inplace=True)
    similarity_scores = cosine_similarity(pt)

    return similarity_scores, books, pt, ratings


@app.route("/")
def index():
    start_time = time.time()
    popular_df = trainPopularityModel()
    end_time = time.time()

    execution_time = end_time - start_time

    return render_template(
        "index.html",
        book_isbn=list(popular_df["ISBN"].values),
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_ratings"].values),
        rating=list(popular_df["avg_rating"].values),
        execution_time=execution_time,
    )


@app.route("/recommend")
def recommend_ui():
    data = {
        "recommended_books": [],
        "execution_time": "none",
    }
    return render_template("recommend.html", data=data)


# @app.route("/recommend_books/<book_title>", methods=["post"])
def recommend(book_title):
    start_time = time.time()
    similarity_scores, books, pt, ratings = trainCollaborativeFiltering()
    end_time = time.time()

    execution_time = end_time - start_time

   
    recommended_books = []

    try:
        index = np.where(pt.index == book_title)[0][0]
        similar_items = sorted(
            list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True
        )[1:5]

        for i in similar_items:
            item = []
            temp_df = books[books["Book-Title"] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates("Book-Title")["ISBN"].values))
            item.extend(
                list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values)
            )
            item.extend(
                list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values)
            )
            item.extend(
                list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values)
            )

            recommended_books.append(item)

        books_df = pd.DataFrame(
            recommended_books,
            columns=["ISBN", "Book-Title", "Book-Author", "Image-URL-M"],
        )
        filtered_ratings = ratings[ratings["ISBN"].isin(books_df["ISBN"])]
        result = pd.merge(books_df, filtered_ratings, on="ISBN")
        result.drop_duplicates("ISBN", inplace=True)
        result_list = result.to_dict(orient="records")

        data = {
            "recommended_books": result_list,
            "execution_time": execution_time,
        }
        return data

    except:
        data = {
            "recommended_books": [],
            "execution_time": None,
        }
        return data


@app.route("/about")
def about():
    return render_template("newabout.html")


@app.route("/add-book", methods=['GET','POST'])
def addBook():

    if request.method == 'POST':
        isbn = request.form['isbn']
        book_title = request.form['book_title']
        book_author = request.form['book_author']
        book_rating = request.form['book_rating']
        book_image = request.files['book_image']

        # Save the image file to static folder
        book_image.save('static/{}'.format(book_image.filename))

        # Create a dictionary with the form data
        book_data = {
            'ISBN': isbn,
            'Book-Title': book_title,
            'Book-Author': book_author,
            'Book-Rating': book_rating,
            'Book-Image': book_image.filename
        }

        # Create a DataFrame from the dictionary
        book_df = pd.DataFrame(book_data, index=[0])

        # Write the DataFrame to a CSV file
        with open('./data/newBook.csv', 'a') as f:
            book_df.to_csv(f, header=f.tell() == 0, index=False)






    addedBooks = pd.read_csv("./data/newBook.csv")

    addedBooks = addedBooks.rename(columns={'Book-Title': 'book_title',
                                                'Book-Author':'book_author',
                                                'Book-Image':'book_image',
                              
                                                })



    addedBooks = addedBooks.to_dict(orient="records")  
    print(addedBooks)
    return render_template("add-book.html", books = addedBooks)


@app.route("/rate-book/<book_isbn>", methods=["POST"])
def rateBook(book_isbn):
    rate_book = []
    with open("./data/ratings.csv", "r", newline="") as read_file:
        # Create a CSV reader object
        reader = csv.reader(read_file)

        # Loop through the rows in the file and update as needed
        rows_list = []
        for row in reader:
            rows_list.append(row)
            if row[1] == book_isbn:
                rate_book.append(row)

    if request.method == "POST":
        new_rating = request.form["rating"]
        isbn_to_rate = request.form["book_isbn"]

        with open("./data/ratings.csv", "w", newline="") as write_file:
            writer = csv.writer(write_file)

            for row in rows_list:
                if row[1] == isbn_to_rate:
                    row[2] = new_rating

                writer.writerow(row)

    return redirect(url_for('index'))



@app.route("/book-detail/<book_isbn>")
def bookDetail(book_isbn):
    books_df = pd.read_csv("./data/books.csv")
    book_details = books_df[books_df["ISBN"] == book_isbn]

    book_details = book_details.rename(columns={'Book-Title': 'book_title',
                                                'Book-Author':'book_author',
                                                'Year-Of-Publication':'year_of_publication',
                                                'Image-URL-S':'image_url_s',
                                                'Image-URL-M':'image_url_m',
                                                'Image-URL-L':'image_url_l',
                                                })


    book_details_list = book_details.to_dict(orient="records")

 
    #book recommendation
    recommended_books = recommend(book_details_list[0]['book_title'])['recommended_books']



    return render_template("book-detail.html", 
                           book_details = book_details_list[0],
                           recommended_books=recommended_books,
                           )



@app.route("/searchBook", methods=['GET'])
def searchBook():
    keyword = request.args.get("keyword")
    books_df = pd.read_csv("./data/books.csv")
    filtered_df = books_df[books_df['Book-Title'].str.lower().str.contains(keyword.lower())].head(50)
    filtered_df = filtered_df.rename(columns={'Book-Title': 'book_title',
                                                'Book-Author':'book_author',
                                                'Year-Of-Publication':'year_of_publication',
                                                'Image-URL-S':'image_url_s',
                                                'Image-URL-M':'image_url_m',
                                                'Image-URL-L':'image_url_l',
                                                })
    filtered_df_list = filtered_df.to_dict(orient="records")

    return render_template("search-result.html", books = filtered_df_list)



if __name__ == "__main__":
    app.run(debug=True)
