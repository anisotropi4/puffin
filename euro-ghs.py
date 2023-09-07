#!/usr/bin/env python3
"""euro-ghs: project GHS population data"""

import datetime as dt
from functools import partial

# os.environ["USE_PYGEOS"] = "0"
import pandas as pd
from pyogrio import read_dataframe, write_dataframe
from shapely import set_precision

from shared.util import density, download_file, get_circle, get_population, log

pd.set_option("display.max_columns", None)
CRS = "EPSG:3034"
CHECKPOINT = False
# CRS = "EPSG:32630"

INPATH = "europa.gpkg"
GHURL = (
    "http://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_STAT_UCDB2015MT_GLOBE_R2019A/V1-2/"
)
GHDIR = "GHS_STAT_UCDB2015MT_GLOBE_R2019A"
GHPOINT = "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_short_pnt"
GHSHAPE = "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2"
OUTPATH = "europa-ghs.gpkg"

set_precision_one = partial(set_precision, grid_size=1.0)
get_population_path = partial(get_population, inpath=INPATH)


START = dt.datetime.now()


def get_ghs(circle, filename):
    """get_ghs:

    input:
      circle:
      filename

    returns:
      GeoSeries

    """
    filepath = f"{GHDIR}/{filename}.gpkg"
    zipfile = f"data/{GHSHAPE}.zip"
    zippath = f"zip://{zipfile}!{filepath}"
    r = read_dataframe(zippath).to_crs(CRS)
    return r.clip(circle).reset_index(drop=True)


def main():
    """main: collates GHS data for 'Europe'""

    returns:
      None
    """
    log("start GHS processing")
    centre, circle = get_circle(INPATH)
    write_dataframe(centre.to_frame("geometry"), OUTPATH, "centre")
    write_dataframe(circle, OUTPATH, "circle")
    download_file(f"{GHURL}/{GHSHAPE}.zip", f"{GHSHAPE}.zip")
    ghs_point = get_ghs(circle, GHPOINT)
    ghs_shape = get_ghs(circle, GHSHAPE)
    outer = read_dataframe(INPATH, layer="outer")
    outer = outer.explode(index_parts=False).reset_index(drop=True)
    outer = outer.clip(circle).reset_index(drop=True)
    write_dataframe(outer, OUTPATH, "outer")
    geometry = outer["geometry"].values
    ix = ghs_point["geometry"].sindex.query(geometry, predicate="contains")
    ghs_point = ghs_point.loc[ix[1]].sort_index().reset_index(drop=True)
    write_dataframe(ghs_point, OUTPATH, "point")
    log("wrote GHS point\t")
    ix = ghs_shape["geometry"].centroid.sindex.query(geometry, predicate="contains")
    ghs_shape = ghs_shape.loc[ix[1]].sort_index().reset_index(drop=True)
    ghs_shape = ghs_shape.dropna(axis=1)
    ghs_shape["density"] = density(ghs_shape, "P15")
    write_dataframe(ghs_shape, OUTPATH, "shape")
    log("wrote GHS shape\t")
    ghs_centroid = ghs_shape.copy()
    ghs_centroid["geometry"] = ghs_shape.centroid
    write_dataframe(ghs_centroid, OUTPATH, "centroid")
    log("wrote GHS centroid")


if __name__ == "__main__":
    main()
