from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# In-memory storage for users
users = {}

# Load and prepare dataset
def load_data():
    data = pd.read_csv('spam_data.csv')
    data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
    return data

# Train the model
def train_model(data):
    X = data['Message']
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    return model

# Initialize model
data = load_data()
model = train_model(data)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to log in first!')
            return redirect(url_for('login'))
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users:
            flash('Username already exists. Please choose another.')
        else:
            users[username] = password
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('You were successfully logged in')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out!')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result = ''
    prob_spam = ''
    prob_ham = ''
    explanation = ''
    sentiment = ''
    
    if request.method == 'POST':
        email = request.form['email']
        result, prob_spam, prob_ham, explanation = predict_spam(email, model)
        sentiment = analyze_sentiment(email)
    
    return render_template('index.html', result=result, prob_spam=prob_spam, prob_ham=prob_ham, explanation=explanation, sentiment=sentiment)

def predict_spam(email, model):
    prob = model.predict_proba([email])[0]
    prediction = model.predict([email])[0]
    prob_spam, prob_ham = prob[1], prob[0]
    
    explanation = "This email is flagged as spam due to certain patterns or keywords." if prediction == 1 else "This email is classified as ham."
    
    return (
        'Spam' if prediction == 1 else 'Ham',
        round(prob_spam * 100, 2),
        round(prob_ham * 100, 2),
        explanation
    )

def analyze_sentiment(email):
    analysis = TextBlob(email)
    sentiment = 'Positive' if analysis.sentiment.polarity > 0 else 'Negative' if analysis.sentiment.polarity < 0 else 'Neutral'
    return sentiment

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    email = request.form['email']
    result = request.form['result']
    feedback = request.form['feedback']
    # Save feedback to a file or database
    with open('feedback.txt', 'a') as f:
        f.write(f"Email: {email}\nResult: {result}\nFeedback: {feedback}\n\n")
    return redirect('/')

@app.route('/visualization')
@login_required
def visualization():
    # Generate a simple bar chart for common spam keywords
    keywords = ['Buy', 'Free', 'Discount', 'Win', 'Limited']
    counts = [100, 90, 80, 70, 60]
    
    plt.figure(figsize=(10, 6))
    plt.bar(keywords, counts, color='orange')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.title('Common Spam Keywords')
    
    # Save the plot to a BytesIO object and encode it as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return render_template('visualization.html', img_data=img_base64)

@app.route('/history')
@login_required
def history():
    try:
        with open('feedback.txt', 'r') as f:
            history = f.readlines()
    except FileNotFoundError:
        history = []
    return render_template('history.html', history=history)

@app.route('/tutorial')
@login_required
def tutorial():
    return render_template('tutorial.html')

if __name__ == '__main__':
    app.run(debug=True)

