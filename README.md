# puffin
Project @worldpop and @ghsl population data onto hierachical layers of hexagons

## Introduction
Place holder that will describe download and procesing of:

## European WorldPop data

This raised the interesting question "where is Europe?"

### Initial approach

Using data from [WorldPop](https://www.worldpop.org/) 2020 grid data was projected onto a Voronoi diagram. The first approach using the property that a Voronoi diagram grid forms a square tile and to boundary creation was by calculating a concave hull of the grid Delaunay triangulation. As the concave hull creates a single exterior, to account for discrete features such as islands, triangulation edge lengths  > 5.0km are filtered and the connected network nodes clustered. Small interior features are then filtered. An exampe visualization of the population density of Albania ![here](https://github.com/anisotropi4/puffin/blob/main/images/albania-01.png "Bounded population density visualisation Albania")

### Where is "Europe"? And where is the centre of "Europe"?
Here is an approximate answer based on a circle projected [EPSG:3034](https://epsg.io/3034) geometry based on the extended European ETRS89 geometry
![here](https://github.com/anisotropi4/puffin/blob/main/images/centre-europe-01.png "Bounded European geography")

Which puts the centre of Europe between Norway and Denmark. Who knew?
![here](https://github.com/anisotropi4/puffin/blob/main/images/centre-europe-02.png "Bounded European geography")
![here](https://github.com/anisotropi4/puffin/blob/main/images/centre-europe-03.png "Bounded European geography")



that and have *finally* got the European parts of Russia and Kazakhstan to render by setting a 

### Global Human Settlement (GHS) Layer - Urban Centre Database

Data from [GHS](https://publications.jrc.ec.europa.eu/repository/handle/JRC115586)


# Data License
Thanks to the following for providing data under permissive license:

* WorldPop for providing data under the Creative Commons Attribution 4.0 International License [CCA4.0](http://creativecommons.org/licenses/by/4.0)
* Global Human Settlement Layer - Urban Centre Database (R2019A) under the European Commission Reuse and Copyright Notice 2019 [here](https://commission.europa.eu/legal-notice_en#copyright-notice)
