#!/usr/bin/env python3
"""euro-hx30: project population density onto two hexagon orientations"""

import datetime as dt
import warnings
from functools import partial

# os.environ["USE_PYGEOS"] = "0"
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, write_dataframe
from shapely import set_precision

from shared.util import (
    append_layer,
    density,
    filter_hexagon,
    get_base_grid,
    get_circle,
    get_grid,
    get_oriented_hex,
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


def main(max_range=6):
    """main: project European population and density data onto a hexagons
    in two orientations

    returns:
      None
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    centre, circle = get_circle(INPATH)
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
