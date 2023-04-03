from flask import Flask, request, render_template, redirect
import pymysql.cursors
import bcrypt

app = Flask(__name__)

# Database connection settings
DB_HOST = "localhost"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASS = "mypassword"

# Home page
@app.route("/")
def home():
    return "Welcome to the home page!"

# Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Get login credentials from form
        username = request.form.get("username")
        password = request.form.get("password")

        # Connect to database
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)

        # Query database for user
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

        # Check password hash
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
            # Password is correct, log in user
            # (You can use Flask-Login or a similar library for this)
            return redirect("/")
        else:
            # Invalid credentials, show error message
            return render_template("login.html", error="Invalid username or password.")

    else:
        # Show login form
        return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
