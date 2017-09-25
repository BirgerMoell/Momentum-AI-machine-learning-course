# Import Flask library
from flask import Flask

# Initialize the app from Flask
app = Flask(__name__)

# Define a route to hello_world function
@app.route('/')
def hello_world():
	return 'Hello World!'

# Run the app on http://localhost:8085
app.run(debug=True,port=8085)
