import shutil
import enchant
import requests
import random
import json
import os
import re
import time
import csv
import concurrent.futures
from urllib.parse import urlencode
from lxml import etree
from bs4 import BeautifulSoup


def rotate_proxy(proxy):
    try:
        r = requests.get('https://httpbin.org/ip', proxies={'http': proxy, 'https': proxy}, timeout=2)
    except:
        pass
    return proxy

def generate_dataset(topic, root_folder, desired_records):
    path = os.path.join(root_folder, topic)
    links_path = os.path.join(root_folder, "links")
    num_records = 0

    # Black list urls that contain these substrings for better results.
    blacklist_url = ["youtube", "netflix", "amazon", "twitter", "pdf"]
    payload = {
        # Query.
        'q': f'{topic}',
        # UULE is used to encode a place or exact location in a value used in a url.
        'uule': 'w+CAIQICIYV2F0ZXJsb28sIE9udGFyaW8gQ2FuYWRh',
    }
    # User agent to show the website which device and browser we are using to connect so that we do not get blocked.
    headers = {
        'User-Agent': random.choice(
            [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/74.0.3729.131 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/74.0.3729.169 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
                'Mozilla/5.0 (Windows NT 10.0; Win 64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/74.0.3729.157 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KTML, like Gecko) Chrome/73.0.3683.103 '
                'Safari/537.36',
            ]
        )
    }

    # proxies = {
    #     'https': 'https://124.121.92.216:80',
    #     'http': 'http://217.97.101.134:80'
    # }
    count = 0

    while True:
        if num_records >= desired_records:
            break

        payload['start'] = count

        # Encode our payload parameters for search.
        params = urlencode(payload)
        url = f'https://www.google.com/search?{params}'

        # Add 2 second sleep to avoid getting blacklisted by google.
        time.sleep(3)
        response = requests.get(url, headers=headers, timeout=2)
        if response.status_code == 429:
            print("Scraper detected")
            break
        soup = BeautifulSoup(response.content, "html.parser")
        dom = etree.HTML(str(soup))

        results = []
        # Gets all the a tags on the google results page with this xpath expression.
        result_elements = dom.xpath('//div[@class="yuRUbf"]/a')

        for element in result_elements:
            test = [x for x in blacklist_url if x in element.attrib['href']]
            if len(test) > 0:
                continue
            results.append(element.attrib['href'])

        links_file = os.path.join(links_path, f'{topic}_links.json')

        with open(links_file, 'a',encoding='utf-8') as f:
            f.write(json.dumps(results, indent=2))

        for url in results:
            if num_records >= desired_records:
                break
            try:
                response = requests.get(url, headers=headers, timeout=5)
            except:
                # catch request exception and keep trying new urls.
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            soup_strings = soup.find_all(text=True)
            # Extract the useful text into an array of words.
            words = extract_text(soup_strings)
            if len(words) < 200:
                # Not enough data captured, skip to next url.
                continue
            filename = os.path.join(path, f'{topic}({num_records}).json')
            with open(filename, 'w+', encoding='utf-8') as f:
                print(f"writing {filename} : {len(words)} words")
                f.write(json.dumps(words, indent=2))
            num_records += 1
        count += 10


def extract_text(soup_string):
    output = ""
    d = enchant.Dict("en_US")
    words_in_dict = 0

    # 75% of the words should in the en_US enchant dict.
    min_allowance = 0.75

    # Black certain tag names from the soup results.
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style',
        'a',
        'footer'
    ]

    # Stop words that are common in text documents. I.e not helpful for text analysis.
    stopwords = ["icon", "external", "plus", "font", "header", "empty", "noscript", "lynx", "br", "flyout", "ff", "text", "react", "div", "a", "about", "above", "after", "again", "against", "all", "am",
                 "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
                 "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
                 "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here",
                 "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                 "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my",
                 "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                 "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than",
                 "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
                 "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
                 "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's",
                 "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
                 "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    for t in soup_string:
        # Only append text from useful elements.
        if t.parent.name == "p":
            text = t.string
            text = re.sub("[^A-Za-z0-9 ]+", " ", text.lower().strip())
            output += f'{text} '
    output = [w for w in output.split() if len(w) > 1 and not w.isdigit() and w not in stopwords and d.check(w)]

    words_in_dict = len([w for w in output if d.check(w)])
    if len(output) == 0:
        return ""
    ratio_words_in_dict = words_in_dict / len(output)
    # Return empty string if min allowance is exceeded.
    if ratio_words_in_dict < min_allowance:
        return ""

    return output


def main():
    #r = requests.get('https://httpbin.org/ip')
    root_folder = 'dataset2'
    topics = ["technology", "gardening", "health"]
    num_examples = 80
    try:
        os.mkdir(root_folder)
    except FileExistsError:
        print(f"Removing {root_folder}")
        # Remove existing data set and replace with new folder.
        shutil.rmtree(root_folder)
        os.mkdir(root_folder)

    os.mkdir(os.path.join(root_folder, "links"))
    for t in topics:
        print(f'gathering documents from search results from {t}...')
        os.mkdir(os.path.join(root_folder, t))
        generate_dataset(t, root_folder, num_examples)


main()
