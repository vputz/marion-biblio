"""
A module for doing quick queries from a CSV input file to JSON query output.
This is meant to be for simple queries where the result will not be
stored separately from the initial CSV files, ie for one-off upload-and-view
queries.

This may change in the future
"""

import csv
import json
import fileinput


def location_and_value( filename, parameters, locationFunc ):
    """
    Returns a dict of
    {
      nodes: { id(int) : { lat: float, lon: float, text: string, value: integer } }
      not_located : [ string ]
    }
    Parameters: dict of

      locationColumn: int
      textColumn: int
      valueColumn: int

    If valueColumn is not defined, a value of 1 is assumed for all locations.
    First row currently not skipped.  This will move to a parameter in future

    In this case, locationfunc maps a single location to either a lat/long pair or None
    """
    result = { 'nodes' : {}, 'not_located' : [] }
    node_index = 0
    with open( filename, encoding="ISO-8859-1") as csvfile:
        reader = csv.reader( csvfile ) #fileinput.FileInput( [filename], openhook=fileinput.hook_encoded("utf-8")) )
        for row in reader :
            try :
                node_index = node_index + 1
                location = row[ int(parameters['locationColumn']) ]
                loc = locationFunc( row[ int(parameters['locationColumn']) ], True )
                if loc == None :
                    result['not_located'].append( location )
                else :
                    node = loc
                    if 'textColumn' in parameters :
                        node['text'] = row[ int(parameters['textColumn']) ]
                    else :
                        node['text'] = ''
                    if 'valueColumn' in parameters :
                        node['value'] = row[ int(parameters['valueColumn']) ]
                    result['nodes'][node_index] = node
            except :
                continue

    return result
