

## Summary 

Side projects solving applications of the Rural Postman Problem using the [postman_problems] library.  Current examples include:

1. 50 States Ride:  Required edges are the state named avenues in DC.  Optional edges are all others roads and trails.  Many more details in the 
accompanying blog post [here][rpp_blog_post].

2. Sleeping Giant: Required edges are the named trails in Sleeping Giant State Park (see [Giant master program]).  Optional edges are the non-required trails and roads.

## 1. 50 States Ride

### Graph

Derived from Open Street Maps data.  Edges are contracted to ease computation.  There are ~400 required edges in the final edgelist.

- required state avenues: **112 miles** 
- connectors between required components: **12 miles**


### Solution

The optimal route covers 160 miles: 124 miles of required road and 36 miles of optional roads and doublebacking.

See the [blog post][rpp_blog_post] for more visualizations and details on the approach. 

[Here][rpp_50states_solution.geojson] is the route (geojson map).

[Here][rpp_50states_solution_csv] is the solution edgelist CSV with lat and lon and street names.



### Contents

- `50states-blogpost.ipynb`: notebook solving and visualizing the problem 
- `rpp_solution.csv`: eulerian circuit solution solution to RPP


## Sleeping Giant

### Graph 
 
For more details on the Sleeping Giant problem and graph, check out the original blogpost [here][Blog post: Intro to Graph Optimization]. 
 
### Solution

The optimal route covers 30.7 miles: 25.6 of required trails and 5.1 miles of optional trails and doublebacking.


## Project

### Contents

- `environment.yml`: environment used to produce `50states-blogpost.ipynb`. 
- `graph.py`: module consolidating heavy lifting for blog post

### Resources

- [postman_problems]: Python package with RPP and CPP solvers
- [Blog post: 50 stats RPP][rpp_blog_post]: solving and visualizing RPP with Leaflet using DC OSM data
- [Blog post: DataCamp]: Original blog post implementing CPP from scratch using Networkx 1.11
- [Blog post: Intro to Graph Optimization]: Same post as above, but modified for Networkx 2.0 



[rpp_50states_solution.geojson]: https://github.com/brooksandrew/50states/blob/master/50states/maps/rpp_solution.geojson
[postman_problems]: https://github.com/brooksandrew/postman_problems

[rpp_50states_solution_csv]: https://github.com/brooksandrew/50states/blob/master/50states/rpp_solution.csv
[Blog post: DataCamp]: https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial
[Blog post: Intro to Graph Optimization]: http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/

[rpp_blog_post]: http://brooksandrew.github.io/simpleblog/articles/fifty-states-rural-postman-problem/
[Giant master program]: http://www.sgpa.org/hikes/masters.html