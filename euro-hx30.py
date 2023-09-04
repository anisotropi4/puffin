#!/usr/bin/env python3
"""euro-hx30: project population density onto two hexagon orientations"""

import datetime as dt
from functools import partial

# os.environ["USE_PYGEOS"] = "0"
import geopandas as gp
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, write_dataframe
from shapely import set_precision

from shared.util import (
    append_layer,
    filter_hexagon,
    get_hex_id,
    get_hexagon,
    get_hx_centre,
    get_population,
    log,
    set_pivot,
)

pd.set_option("display.max_columns", None)
CRS = "EPSG:3034"
CHECKPOINT = False
# CRS = "EPSG:32630"

INPATH = "europa.gpkg"
OUTPATH = "europa-hex.gpkg"

set_precision_one = partial(set_precision, grid_size=1.0)
get_population_path = partial(get_population, inpath=INPATH)


START = dt.datetime.now()


def get_circle():
    """get_circle:

    returns:
      circle GeoDataFrame centre Poinr and Polygon
    """
    circle = read_dataframe(INPATH, layer="circle")
    centre = circle.centroid
    return centre, circle


def get_grid(n, circle):
    """get_grid: return sufficient grid points to allow construction of hexagons"
    "in two orientations to cover circle. boost and offset parameters determined"
    "by trial and error"""
    hex_id = get_hex_id(n, circle)
    r = gp.GeoSeries(get_hx_centre(hex_id), crs=CRS)
    return r.apply(set_precision_one)


def density(this_gf, column="population"):
    """density: from GeoSeries 'this_gf' return km2 density of 'column'"""
    return 1.0e6 * this_gf[column] / this_gf.area


def get_base_grid(n, circle):
    """get_base_grid:

    input:
      n: h3 level
      circle: perimiter

    returns:
      GeoSeries hexagon point grid
    """
    circle_zone = circle.buffer(370.0e3)
    if n == 1:
        circle_zone = circle.buffer(600.0e3)
    return get_grid(n, circle_zone)


def get_oriented_hex(angle, base_grid, pivot, circle):
    """get_oriented_hx:

    input:
      angle:
      base_grid:
      privot:
      circle:

    returns:
      rotated GeoSeries hexagon grid clipped to circle boundary
    """
    r = get_hexagon(base_grid.rotate(angle, origin=pivot))
    r = r.clip(circle).explode(index_parts=False)
    ix = r.type == "Polygon"
    return r[ix].reset_index(drop=True)


def main(max_range=6):
    """main:

    returns:
      None
    """
    centre, circle = get_circle()
    write_dataframe(centre.to_frame("geometry"), OUTPATH, "centre")
    write_dataframe(circle, OUTPATH, "circle")
    outer = read_dataframe(INPATH, layer="outer")
    outer = outer.explode(index_parts=False).reset_index(drop=True)
    outer = outer.clip(circle).reset_index(drop=True)
    write_dataframe(outer, OUTPATH, "outer")
    outer = outer["geometry"]
    base_grid = get_grid(3, circle)
    pivot = set_pivot(OUTPATH, centre, base_grid)
    for n in range(1, max_range):
        base_grid = get_base_grid(n, circle)
        for m in ["00", "30"]:
            log(f"{str(n).zfill(2)}\thex world-pop")
            log(f"{str(n).zfill(2)}\t{m}")
            key = f"{str(n).zfill(2)}-{m}"
            hexagon = get_oriented_hex(float(m), base_grid, pivot, circle)
            layer = f"hexagon-{key}"
            write_dataframe(hexagon.to_frame("geometry"), OUTPATH, layer)
            layer = f"interpolate-{key}"
            if append_layer(OUTPATH, layer):
                continue
            r = filter_hexagon(hexagon, outer)
            step = np.ceil(np.log2(r.size)).astype(int)
            for i, j in enumerate(np.array_split(r.index, step)):
                log(f"{str(i+1).zfill(5)} {str(step).zfill(5)}\thx")
                if CHECKPOINT:
                    write_dataframe(r, OUTPATH, layer=layer)
                r.loc[j, "population"] = r.loc[j, "geometry"].map(get_population_path)
            r["density"] = density(r)
            write_dataframe(r.reset_index(drop=True), OUTPATH, layer=layer)
            log(f"wrote layer {key}")
    log("wrote hex world-pop")


if __name__ == "__main__":
    main()
