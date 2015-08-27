"""
A module for reading Web of Science tab-delimited (UTF-8, Windows) files

A key to the WoK fields,
http://images.webofknowledge.com/WOKRS57B4/help/WOS/hs_wos_fieldtags.html
FN File name
VR Version Number
PT Publication Type (J = journal B = book, S = Series, P=Patent)
AU Authors last name/initial
AF Authors, full name
BA Book Authors
BF Book authors, full name
CA Group Authors
GP Book Group Authors
BE Editors
TI title
SO Source (journal)
SE Series title
LA language
DT Document Type
CT Conferency Title
CY Conference Date
CL Conference Location
SP Conference Sponsors
HO Conference Host
DE Author Keywords
ID KeyWords Plus
AB abstract
C1 Addresses (of authors)
RP Reprint Address
EM Email contact
FU Funding and grant number
FX Funding text
CR Cited References
NR Cited Reference Count
TC WOS Times CIted count
Z9 Total times cited count (WoS, BCI, CSCD)
PU Publisher
PI Publisher City
PA Publisher Address
SN International Standard Serial Number (ISSN)
BN International Standard Book Number (ISBN)
J9 29-char source abbreviation
JI ISO source abbreviation
PD Publication date
PY Year Published
VL Volume
IS Issue
SI Special issue
PN Part Number
SU Supplement
MA Meeting Abstract
BP Beginning Page
EP Ending Page
AR Article Number
DI Digital Object Identifier (DOI)
D2 Book digital object identifier (DOI)
PG Page count
P2 Chapter Count
WC Web of Science categories
SC research areas
GA document delivery number
UT accession number
ER end of record
EF end of file
"""
import csv
import fileinput
from collections import Counter
import logging
import itertools
import re
from marion_biblio import progressivegenerators


def authorlist_from_authorfield(string):
    """Parse the author field, returning a list of authors"""
    return [x.strip() for x in string.split(";")]


def addresslist_from_addressfield(string):
    """Parse the address field, returning a list of addresses"""
    return [x.strip() for x in string.split(";")]

flatten_chain = itertools.chain.from_iterable

COUNTRY_WORDS = [x for x in """
Albania
Algeria
Arab Emirates
Argentina
Armenia
Australia
Austria
Azerbaijan
Bahamas
Bangladesh
Belgium
Benin
Bolivia
Brazil
Bulgaria
Byelarus
Cambodia
Cameroon
Canada
Chile
China
Colombia
Combodia
Costa Rica
Cote Ivoire
Croatia
Cuba
Cyprus
Czech Republic
Denmark
Ecuador
Egypt
England
Estonia
Ethiopia
Faso
Finland
France
Georgia
Germany
Ghana
Greece
Guatemala
Hungary
Iceland
India
Indonesia
Iran
Ireland
Israel
Italy
Japan
Jordan
Kenya
Kuwait
Kyrgyzstan
Latvia
Lebanon
Liberia
Lithuania
Luxembourg
Macedonia
Malaysia
Mali
Malta
Mexico
Moldova
Monaco
Monteneg
Montenegro
Morocco
Netherlands
New Zealand
Nigeria
North Korea
Norway
Oman
Pakistan
Paraguay
Peru
Poland
Polynesia
Portugal
Qatar
Rep Congo
Romania
Russia
Saudi Arabia
Scotland
Serbia
Singapore
Slovakia
Slovenia
South Africa
South Korea
Spain
Spain
Sri Lanka
Sweden
Switzerland
Taiwan
Tanzania
Thailand
Tunisia
Turkey
Uganda
Ukraine
Uruguay
Usa
Uzbekistan
Venezuela
Vietnam
Wales
Zambia
""".split("\n") if len(x) > 0]

COUNTRY_REGEXES = [re.compile(x+"$") for x in COUNTRY_WORDS]


def country_from_address(s, countries=COUNTRY_REGEXES):
    """Tries to match a country to the last few words of an address"""
    # strip semicolon and trailing space, lowercase, and capitalize each word
    s = s.strip("; ").lower().title()
    # match to regexes in country_list
    for x in countries:
        m = x.search(s)
        if m is not None:
            return m.group(0)
    # if no match, print unmatched last two words (to build word list)
    print("Not found: "+str(s.split()[-2:]))
    return "Error"


def countrylist_from_addresses(addrs):
    """ Creates a list of countries from a string of addresses """
    # break on [author] sections
    strings = [x for x in flatten_chain(
        [x.split(';') for x in re.split("\[.*?\]", addrs)])
        if len(x.strip()) > 0]
    # print strings
    return list(set([country_from_address(x) for x in strings if len(x) > 0]))


def dict_from_addresses(addrs):
    """ creates an author dict from addresses """

    # first split on semicolons outside brackets
    rsplit = re.compile(r"; (?=\[)")
    s2 = rsplit.split(addrs)
    result = {}
    for s3 in s2:
        r = re.compile("\[(.*?)\]\s*(.*?)$")
        pairs = r.findall(s3)
        for p in pairs:
            names = [x.strip() for x in p[0].split(';')]
            for name in names:
                if name in result:
                    result[name] = result[name] + "; " + p[1].strip()
                else:
                    result[name] = p[1].strip()
    return result


def cited_dois(item):
    """parse cited documents and return a list of cited DOIs"""
    return re.findall('DOI\s*([^\s;]*)', item['CR'])


class WosReader(object):

    def __init__(self, files):
        self.files = files

    def reader(self):
        return csv.DictReader(
            fileinput.FileInput(self.files,
                                openhook=fileinput.hook_encoded("utf-8")),
            dialect='excel-tab')

    def fields(self, field):
        return [row[field] for row in self.reader()]

    def wordle_strings(self, field="SC"):
        return flatten_chain([[x.strip() for x in row[field].split(";")]
                              for row in self.reader()])

    def wordle_string(self, field="SC"):
        ws = self.wordle_strings(field)
        wsc = Counter(ws)
        results = []
        for k, v in wsc.iteritems():
            results.append(k + ":" + str(v))
        return "\n".join(results)

    def sources_counter(self):
        return Counter([row['JI'] for row in self.reader()])

    def authors_counter(self):
        return Counter(flatten_chain([authorlist_from_authorfield(row['AU'])
                                      for row in self.reader()]))

    def set_authors(self):
        return set(self.authors_counter().iterkeys())

    def countries_counter(self):
        return Counter(flatten_chain([countrylist_from_addresses(x)
                                      for x in self.address_strings()]))

    def address_strings(self):
        return [row['C1'] for row in self.reader()]

    def set_cited_dois(self):
        return list(set(flatten_chain((cited_dois(x) for x in self.reader()))))

    def dois(self):
        return [x['DI'] for x in self.reader() if x['DI'] != '']

    def set_years(self):
        return set((x['PY'] for x in self.reader() if x['PY'].isdigit()
                    and int(x['PY']) > 1900))

    def by_year_iterator(self, py):
        return (x for x in self.reader() if x['PY'] == py)

    def country_count_by_year(self):
        yl = list(self.set_years())
        yl.sort(lambda a, b: cmp(int(a), int(b)))
        result = {}
        for year in yl:
            result[year] = Counter(
                flatten_chain([countrylist_from_addresses(x)
                               for x in [row['C1']
                                         for row in self.by_year_iterator(year)]]))
        return result

    def padded_country_count_by_year(self, top_x=7):
        yl = list(self.set_years())
        cyr = self.country_count_by_year()
        country_counter = self.countries_counter()
        l = country_counter.items()
        l.sort(lambda a, b: cmp(b[1], a[1]))

        countries = [x[0] for x in l[0:top_x]]
        Result = {}
        for year in cyr.iterkeys():
            Result[year] = {}
            for country in countries:
                if country in cyr[year]:
                    Result[year][country] = cyr[year][country]
                else:
                    Result[year][country] = 0
        return Result


def download_names(stem, lastnum):
    return [stem+".tab"] + [stem+"_("+str(x)+").tab"
                            for x in range(1, lastnum+1)]


from numpy import array
import tables


class Paper(tables.IsDescription):
    index = tables.Int32Col()  # index for authors, etc
    doi = tables.StringCol(50)  # max in one dump was 31 DI
    title = tables.StringCol(500)  # max in one was 174 TI
    journal = tables.StringCol(30)  # J9
    pubdate = tables.StringCol(8)  # PD
    pubyear = tables.Int16Col()  # PY
    # max author was 22


class Author(tables.IsDescription):
    author = tables.StringCol(40)
    address = tables.StringCol(255)
    paper_index = tables.Int32Col()


class Keyword(tables.IsDescription):
    keyword = tables.StringCol(80)


def make_pytable(w, filename, title="test",
                 progressReporter=progressivegenerators.NPG):
    """parses the wos reader and converts everything
    into an HDF5 pytable for faster access"""
    def create_skeleton(filename):
        h5file = tables.open_file(filename, mode='w', title=title)
        table = h5file.create_table(
            h5file.root, 'papers', Paper, 'WOS paper records')
        authors = h5file.create_vlarray(h5file.root, 'authors',
                                        tables.StringAtom(40))
        addresses = h5file.create_vlarray(
            h5file.root, 'addresses', tables.StringAtom(60))
        countries = h5file.create_vlarray(h5file.root, 'countries',
                                          tables.StringAtom(30))
        cited_papers = h5file.create_vlarray(h5file.root, 'cited_papers',
                                             tables.StringAtom(50))
        abstracts = h5file.create_vlarray(h5file.root, 'abstracts',
                                          tables.VLStringAtom())
        categories = h5file.create_vlarray(h5file.root, 'categories',
                                           tables.StringAtom(40))
        authortable = h5file.create_table(h5file.root, 'authortable',
                                          Author, "WOS Author data")
        return h5file

    def append_paper(h5file, index, paper_entry):
        result = h5file.root.papers.row
        result['index'] = index
        result['doi'] = paper_entry['DI']
        result['title'] = paper_entry['TI']
        result['journal'] = paper_entry['J9']
        result['pubdate'] = paper_entry['PD']
        result['pubyear'] = int(paper_entry['PY']) \
                            if paper_entry['PY'].isdigit() \
                               else -1
        result.append()

    def append_authors(h5file, paper_entry):
        h5file.root.authors.append(authorlist_from_authorfield(paper_entry['AU']))

    def is_valid_paper_entry(paper_entry):
        fields = 'DI TI J9 PD PY AU AF C1 AB'.split()
        return False not in [paper_entry[f] is not None for f in fields] \
            and paper_entry['DI'] != 'DI' \
                                     and paper_entry['DI'] != b'' \
                                     and paper_entry['DI'] != ''

    def append_categories(h5file, paper_entry):
        try:
            h5file.root.categories.append(authorlist_from_authorfield(paper_entry['WC']))
        except AttributeError as e:
            print(paper_entry['WC'])
            print(paper_entry)
            raise(e)

    def append_to_authortable(h5file, paper_index, paper_entry):
        # now if each author is not already in the table,
        # add to the author table
        aulist = authorlist_from_authorfield(paper_entry['AU'])
        aflist = authorlist_from_authorfield(paper_entry['AF'])
        if len(aulist) != len(aflist):
            print("ERROR, AUTHOR LISTS DIFFERENT LENGTH")
            return
        addir = dict_from_addresses(paper_entry['C1'])
        for author, address in zip(aulist, aflist):
            # print matches
            newauthor = h5.root.authortable.row
            newauthor['author'] = author
            newauthor['address'] = addir.get(address, "Not found")
            newauthor['paper_index'] = paper_index
            newauthor.append()
            h5file.root.authortable.flush()
        h5file.root.countries.append(countrylist_from_addresses(p['C1']))
        h5file.root.cited_papers.append(cited_dois(p))
        h5file.root.abstracts.append(p['AB'])

    def flush_tables(h5file):
        h5file.root.papers.flush()
        h5file.root.authortable.flush()
        h5file.root.authortable.cols.author.create_index()
        h5file.root.authortable.cols.paper_index.create_index()
        h5file.root.authortable.flush()
        
    h5 = create_skeleton(filename)

    index = 0
    for p in progressivegenerators.reporterProgressGenerator(
            w.reader(), progressReporter):

        if is_valid_paper_entry(p):
            append_paper(h5, index, p)
            append_authors(h5, p)
            append_categories(h5, p)
            append_to_authortable(h5, index, p)
            index = index + 1
        else:
            logging.info("No DOI, paper ignored: {0}".format(p['TI']))

    flush_tables(h5)
    h5.close()


class Wos_h5_reader():

    def __init__(self, filename):
        self.h5 = tables.openFile(filename, 'r')

    def all_dois(self):
        return array([x['doi'] for x in self.h5.root.papers if x['doi'] != ''])

    def countries_counter(self):
        return Counter(flatten_chain([x for x in self.h5.root.countries]))

    def addresses_from_paper(self, index):
        return [x['address'] for x in self.h5.root.authortable.where(
            'paper_index == ' + str(index))]

    def all_cited_dois(self):
        return array(list(set(
            flatten_chain([x for x in self.h5.root.cited_papers]))))

    def all_authors(self):
        return array(list(set(
            flatten_chain([x for x in self.h5.root.authors]))))

    def all_title_words(self):
        return list(set(itertools.chain.from_iterable(
            (words(x['title']) for x in self.h5.root.papers))))

    def all_title_stems(self):
        return list(set(itertools.chain.from_iterable(
            (stems(x['title']) for x in self.h5.root.papers))))

    @property
    def papers(self):
        return self.h5.root.papers

    def dict_doi_to_authors(self):
        Result = {}
        for paper in self.h5.root.papers:
            Result[paper['doi']] = self.h5.root.authors[paper['index']]
        return Result

from contextlib import contextmanager


@contextmanager
def open_wos_h5(filename):
        w5 = Wos_h5_reader(filename)
        yield (w5)
        w5.h5.close()


@contextmanager
def open_wos_tab(filename):
    wos = WosReader(filename)
    yield(wos)


# make_pytable( WosReader("metamaterials_cited.tab"),
# "metamaterials.h5", "Metamaterials" )
# w5 = Wos_h5_reader( "metamaterials.h5" )
w = WosReader(["switz_oct13_1.csv", "switz_oct13_2.csv", "switz_oct13_3.csv"])
