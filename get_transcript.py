import urllib.request
import re
import json
import sys
import os
import time

lim = 10
start = 0
cd_count = 5
cd_time = 5
crawl = True

if len(sys.argv) < 2:
    raise('Please specify language')
else:
    label = sys.argv[1]
    if len(sys.argv) > 2:
        lim = int(sys.argv[2])
        if len(sys.argv) > 3:
            start = int(sys.argv[3])


top_url = 'https://www.ted.com'
root_pages = {'tw':'https://www.ted.com/talks?language=zh-tw&sort=popular',
    'cn':'https://www.ted.com/talks?language=zh-cn&sort=popular',
    'jp':'https://www.ted.com/talks?language=ja&sort=popular',
    'kr':'https://www.ted.com/talks?language=ko&sort=popular'}

codes = {'tw':'zh-tw', 'cn':'zh-cn', 'jp': 'ja', 'kr': 'ko'}


root_page = root_pages[label]

all_links = []


count = 0
page = 0

page_url = root_page

if crawl:
    while count < lim:
        prev_count = count
        page = page + 1
        page_url = root_page + '&page=' + str(page)
        try:
            html = urllib.request.urlopen( page_url  ).read()
        except urllib.error.HTTPError as e:
            print(e)
            print(e.code)
            if e.code == 404 or e.code == '404':
                print('last page reached')
                break
            else:
                print('longer cooling down......')
                time.sleep(10*cd_time)
                page = page - 1
                continue


        links = re.findall(r"href=\\'(\/talks\/[\S\?\-]+)\\'", str(html))
        all_links += set(links)
        count = len( all_links )
        print('now ' + str(count) + ' links in the database')
        if count == prev_count:
            break
        if page % cd_count == 0:
            print('cooling down......')
            time.sleep(cd_time)

    count = len( all_links )
    print('got ' + str(count) + ' links in the database')

    all_links = list(set(all_links))
    count = len( all_links )
    print('got ' + str(count) + ' UNIQUE links in the database')


    with open(label + '_ted_database.json', 'w') as outfile:
        json.dump(all_links, outfile, ensure_ascii=False, indent=2)

else:
    json_file = open(label + '_ted_database.json', 'r')
    all_links = json.load(json_file)
    all_links = all_links[start:]


dest_path = 'data/' + label +'/ted/'

if os.path.exists(dest_path):
    pass
else:
    os.mkdir(dest_path)

retry_links = []

count = 0
for link in all_links:
    filename = re.findall(r"\/talks\/([\S\?\-]+)\?", link)[0]
    print(filename)
    transcript_url = top_url + '/talks/' + filename \
        + '/transcript.json?language=' + codes[label]
    print(transcript_url)

    try:
        jsn = urllib.request.urlopen(transcript_url)
    except urllib.error.HTTPError as e:
        print(e)
        print(e.code)
        if e.code == 404 or e.code == '404':
            print('page not found')
            continue
        else:
            print('longer cooling down......')
            time.sleep(10*cd_time)
            all_links.append( link )
            retry_links.append( link )
            continue

    jsn = jsn.read().decode(jsn.headers.get_content_charset())

    jsn2 = str( jsn )
    jsn2 = json.loads( jsn2 )

    with open(dest_path + filename + '.json', 'w') as outfile:
        json.dump(jsn2, outfile, ensure_ascii=False, indent=2)
    count = count + 1
    if count % cd_count == 0:
        print('cooling down......')
        time.sleep(cd_time)

print('downloaded ' + str(count) + ' jsons')
print(retry_links)
