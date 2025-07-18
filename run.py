# run.py
import webbrowser
import threading
import time
from app import app

def open_browser():
    # give the server a second to start up
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # start the browser in a separate thread
    threading.Thread(target=open_browser).start()

    print("Starting server at http://127.0.0.1:5000")
    # this will keep the console open and serve your app
    app.run(host="127.0.0.1", port=5000)
