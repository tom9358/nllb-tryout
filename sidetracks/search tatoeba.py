import requests

q = 'wiend'

url = "https://api.tatoeba.org/unstable/sentences"
params = {
    "lang": "gos",
    "q": q, #search in tatoeba database here
    "include_unapproved": "yes"
}

response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print('success, found',len(data['data']),'\n')
    #print(data)  # Print the response JSON
else:
    print(f"Request failed with code: {response.status_code}")

for i in data['data']:
    print(i['text'])