"""This module is intended to look up information about a DOI by
using crossref.org's generous guest access.

"""

import urllib.request
import json


def doi_info(doi):
    udoi = doi if isinstance(doi, str) else doi.decode('utf-8')
    url = 'http://api.crossref.org/works/'+udoi
    with urllib.request.urlopen(url) as response:
        result = json.loads(response.read().decode('utf-8'))['message']
    return result


def doi_info_or_none(doi):
    try:
        result = doi_info(doi)
    except:
        result = {'title': ["NOT FOUND"],
                  'publisher': "NOT FOUND",
                  'author': [{'family': 'NOT FOUND',
                              'given': 'NOT FOUND',
                              'affiliation': 'NOT FOUND'}],
                  'container-title': ['NOT FOUND']}
    return result


def authors_from_info(i):
    return [" ".join([a.get('given', ''), a.get('family', '')]).strip()
            for a in i['author']]


def title_from_info(i):
    if len(i['title']) > 0:
        return i['title'][0]
    else:
        return 'TITLE NOT FOUND'
