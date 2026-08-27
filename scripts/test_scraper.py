import os
import sys
import json
import re
import urllib.request
from datetime import datetime

def extract_all_wrecks():
    url = "https://en.wikipedia.org/wiki/List_of_shipwrecks_in_the_Indian_Ocean"
    req = urllib.request.Request(url, headers={'User-Agent': 'EchoPulseNet/2.6 (Marine Sonar Intelligence)'})
    html = urllib.request.urlopen(req, timeout=12).read().decode('utf-8')

    # Regex to grab each <tr> block
    tr_blocks = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    print(f"Total <tr> blocks found: {len(tr_blocks)}")

    wrecks = []
    for tr in tr_blocks:
        # Find decimal coordinates in geo-dec or geo
        # Pattern: 21.350°N 59.967°E or geo-dec">13.50°N 48.93°E or <span class="geo">20.27722; 70.99361</span>
        lat = None
        lon = None

        geo_span = re.search(r'<span class="geo">([0-9\.\-]+);\s*([0-9\.\-]+)</span>', tr)
        if geo_span:
            lat = float(geo_span.group(1))
            lon = float(geo_span.group(2))
        else:
            dec_match = re.search(r'class="geo-dec"[^>]*>([0-9\.\-]+)°([NS])\s+([0-9\.\-]+)°([EW])', tr)
            if dec_match:
                lat_val = float(dec_match.group(1))
                if dec_match.group(2) == 'S':
                    lat_val = -lat_val
                lon_val = float(dec_match.group(3))
                if dec_match.group(4) == 'W':
                    lon_val = -lon_val
                lat = lat_val
                lon = lon_val

        if lat is not None and lon is not None:
            # Extract ship name
            name_m = re.search(r'<a [^>]*title="([^"]+)"', tr)
            ship_name = name_m.group(1) if name_m else "Historic Maritime Shipwreck"
            # Extract year/date if available
            date_m = re.search(r'<td>(\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4})</td>', tr)
            sunk_date = date_m.group(1) if date_m else "Historical Period"
            
            # Clean text notes
            clean_text = re.sub(r'<[^>]+>', ' ', tr)
            clean_text = ' '.join(clean_text.split())

            wrecks.append({
                "name": ship_name,
                "latitude": lat,
                "longitude": lon,
                "date": sunk_date,
                "description": clean_text
            })

    print(f"Extracted {len(wrecks)} shipwrecks with precise coordinates.")
    return wrecks

if __name__ == "__main__":
    w = extract_all_wrecks()
    for item in w:
        print(f"-> {item['name']} ({item['latitude']}, {item['longitude']}) - {item['date']}")
