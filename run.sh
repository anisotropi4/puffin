#!/usr/bin/env bash

if [ ! -d venv ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade wheel
    if [ -s requirements.txt ]; then
        pip install --upgrade -r requirements.txt | tee setup.txt
    fi
fi

for i in archive data output
do
    if [ ! -d ${i} ]; then
        mkdir ${i}
    fi
done

source venv/bin/activate

echo get population data
if [ ! -s europa.gpkg ]; then
    ./get-tiff-europe.py
fi

echo process population data
if [ ! -s europa-hex.gpkg ]; then
    ./euro-hx30.py
fi

echo process GHSL data
if [ ! -s europa-ghs.gpkg ]; then
    ./euro-ghs.py
fi
