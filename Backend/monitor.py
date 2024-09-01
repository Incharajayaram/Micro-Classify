import requests
import time

def monitor_server(url="http://localhost:8000"):
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Server is up: {response.status_code}")
            else:
                print(f"Server responded with status code: {response.status_code}")
        except requests.ConnectionError:
            print("Server is down!")
        time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    monitor_server()
