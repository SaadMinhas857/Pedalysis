# time_utils.py
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def increment_time(current_time, frames, fps):
    """ Increment the given time by the number of frames processed at the specified fps. """
    return current_time + timedelta(seconds=frames / fps)

def get_time_from_xml(xml_file):
    """ Extract the starting time from an XML file. """
    namespaces = {
        'ns': 'urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.00',
        'lib': 'urn:schemas-professionalDisc:lib:ver.2.00',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    tree = ET.parse(xml_file)
    root = tree.getroot()
    creation_date = root.find('ns:CreationDate', namespaces)
    if creation_date is None:
        raise ValueError("CreationDate tag not found in XML.")
    timestamp = creation_date.attrib.get('value')
    if timestamp is None:
        raise ValueError("Timestamp 'value' attribute not found in CreationDate tag.")
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')