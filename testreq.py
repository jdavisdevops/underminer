import requests

pat = "65c0aa487ddc1e1b83241e6608c7316906f8a35d"

url = r"https://api-sandbox.uphold.com/v0/me"

headers = {
    "User-Agent": None,
    "Accept-Encoding": None,
    "Connection": None,
    # "Content-Type": "application/json",
    "Authorization": f"Bearer {pat}",
}

r = requests.get(url, headers=headers)

print(r.text)