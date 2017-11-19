

## Summary 

Side project solving an application of the Rural Postman Problem on a graph constructed from roads in DC. 
Required edges are the state named avenues; optional edges are all others roads and trails.  Many more details in the 
accompanying blog post [here][rpp_blog_post].


## Graph

Derived from Open Street Maps data.  Edges are contracted to ease computation.  There are ~400 required edges in the final edgelist.

- required state avenues: **112 miles** 
- connectors between required components: **12 miles**


## Solution

The optimal route covers 160 miles: 124 miles of required road and 36 miles of optional roads and doublebacking.

[Here][rpp_solution.geojson] is the route:



## Contents

- `50states-blogpost.ipynb`: notebook solving and visualizing the problem 
- `graph.py`: module consolidating heavy lifting for blog post
- `rpp_solution.csv`: eulerian circuit solution solution to RPP
- `environment.yml`: environment used to produce `50states-blogpost.ipynb`. 

## Resources

- [postman_problems]: Python package with RPP and CPP solvers
- [Blog post: DataCamp]: Original blog post implementing CPP from scratch using Networkx 1.11.
- [Blog post: Intro to Graph Optimization]: Same post as above, but modified for Networkx 2.0 


[rpp_solution.geojson]: https://github.com/brooksandrew/50states/blob/master/rpp_solution.geojson
[postman_problems]: https://github.com/brooksandrew/postman_problems
[]:http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/

[Blog post: DataCamp]: https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial
[Blog post: Intro to Graph Optimization]: http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
[rpp_blog_post]: http://brooksandrew.github.io/simpleblog/fifty-states-rural-postman-problem/