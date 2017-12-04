/*

made based on tutorial by ZevRoss
http://zevross.com/blog/2014/09/30/use-the-amazing-d3-library-to-animate-a-path-on-a-leaflet-map/

*/


var token = 'pk.eyJ1IjoibGF1cmEzNzYiLCJhIjoiY2o5dnM2M2htMXB1ejJwcG94NXdpbm5qaSJ9.vcrHmCTIsE7wdBIksd2WTQ';
var mapId = 'mapbox.mapbox-streets-v7';
var style = 'mapbox://styles/mapbox/outdoors-v9';
var geoData = 'https://gist.githubusercontent.com/brooksandrew/c71508bcf67335df1c379ac7decec2e7/raw/fe6a7cf6579376c2e8696c3876b1fd9637438d2d/sleeping_giant_osm_rpp.geojson';


var tileLayer = L.tileLayer('https://{s}.tiles.mapbox.com/v4/{mapId}/{z}/{x}/{y}.png?access_token={token}', {
  attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
  subdomains: ['a','b','c','d'],
  mapId: mapId,
  token: token
});

var map = L.map('map')
  // .addLayer(style)
  .addLayer(tileLayer)
  .setView([41.431341, -72.88], 16);


map.on('load', function () {

    map.addLayer({
        "id": "terrain-data",
        "type": "line",
        "source": {
            type: 'vector',
            url: 'mapbox://mapbox.mapbox-terrain-v2'
        },
        "source-layer": "contour",
        "layout": {
            "line-join": "round",
            "line-cap": "round"
        },
        "paint": {
            "line-color": "#ff69b4",
            "line-width": 1
        }
    });
});



/****************** SVG elements *******************/

var svg = d3.select(map.getPanes().overlayPane).append("svg");
var g = svg.append("g").attr("class", "leaflet-zoom-hide");

d3.json(geoData, function(collection){

  // transform geojson to path
  var transform = d3.geo.transform({
    point: projectPoint
  })
  var d3path = d3.geo.path().projection(transform);

  function projectPoint(x, y) {
    var point = map.latLngToLayerPoint(new L.LatLng(y, x));
    this.stream.point(point.x, point.y);
  }

  // project points to a line
  var toLine = d3.svg.line()
    .interpolate("linear")
    .x(function(d) {
      return applyLatLngToLayer(d).x
    })
    .y(function(d) {
      return applyLatLngToLayer(d).y
    });

  function applyLatLngToLayer(d) {
    var y = d.geometry.coordinates[1]
    var x = d.geometry.coordinates[0]
    return map.latLngToLayerPoint(new L.LatLng(y, x))
  }

  var linePath = g.selectAll(".lineConnect")
              .data([collection.features])
              .enter()
              .append("path")
              .attr("class", "lineConnect");

  // This will be our traveling circle it will
  // travel along our path
  var marker = g.append("circle")
      .attr("r", 3)
      // .attr("r", 10)
      .attr("id", "marker")
      .attr("class", "travelMarker");

  // hidden waypoints
  var ptFeatures = g.selectAll("circle")
      .data(collection.features)
      .enter()
      .append("circle")
      .attr("r", 3)
      .attr("class", function(d) {
        return "waypoints" + "c" + d.properties.time
      })
      .style("stroke", function(d) {
        return d.properties.color;
      })
      .style("fill", function(d) {
        // console.log(d.properties.color);
        return d.properties.color;
      })
      .style("opacity", 0); // change opacitiy to 1 to see all point color coded

  var originANDdestination = [collection.features[0], collection.features[collection.features.length - 1]];
  var begend = g.selectAll(".drinks")
      .data(originANDdestination)
      .enter()
      .append("circle", ".drinks")
      .attr("r", 5)
      .style("fill", "red")
      .style("opacity", "1");

  map.on("viewreset", reset);

  reset();
  transition();

  // Reposition the SVG to cover the features.
  function reset() {
      var bounds = d3path.bounds(collection),
          topLeft = bounds[0],
          bottomRight = bounds[1];
      // console.log(bounds);

      // here you're setting some styles, width, heigh etc
      // to the SVG. Note that we're adding a little height and
      // width because otherwise the bounding box would perfectly
      // cover our features BUT... since you might be using a big
      // circle to represent a 1 dimensional point, the circle
      // might get cut off.
      // text.attr("transform",
      //     function(d) {
      //         return "translate(" +
      //             applyLatLngToLayer(d).x + "," +
      //             applyLatLngToLayer(d).y + ")";
      //     });
      // for the points we need to convert from latlong
      // to map units
      begend.attr("transform",
          function(d) {
              return "translate(" +
                  applyLatLngToLayer(d).x + "," +
                  applyLatLngToLayer(d).y + ")";
          });
      ptFeatures.attr("transform",
          function(d) {
              return "translate(" +
                  applyLatLngToLayer(d).x + "," +
                  applyLatLngToLayer(d).y + ")";
          });
      //  harding coding the starting point
      marker.attr("transform",
          function() {
            var y = collection.features[0].geometry.coordinates[1]
            var x = collection.features[0].geometry.coordinates[0]
            return "translate(" +
              map.latLngToLayerPoint(new L.LatLng(y, x)).x + "," +
              map.latLngToLayerPoint(new L.LatLng(y, x)).y + ")";
          });

      // Setting the size and location of the overall SVG container
      svg.attr("width", bottomRight[0] - topLeft[0] + 120)
          .attr("height", bottomRight[1] - topLeft[1] + 120)
          .style("left", topLeft[0] - 50 + "px")
          .style("top", topLeft[1] - 50 + "px");
      // linePath.attr("d", d3path);

      linePath.attr("d", toLine)
        .style("stroke", 'yellow') // THIS IS WHERE TO CHANGE PATH COLOR --- NEED TO ACCESS GEOJSON
        .style("opacity", '0.5')


      // ptPath.attr("d", d3path);
      g.attr("transform", "translate(" + (-topLeft[0] + 50) + "," + (-topLeft[1] + 50) + ")");
  } // end reset

  // the transition function could have been done above using
  // chaining but it's cleaner to have a separate function.
  // the transition. Dash array expects "500, 30" where
  // 500 is the length of the "dash" 30 is the length of the
  // gap. So if you had a line that is 500 long and you used
  // "500, 0" you would have a solid line. If you had "500,500"
  // you would have a 500px line followed by a 500px gap. This
  // can be manipulated by starting with a complete gap "0,500"
  // then a small line "1,500" then bigger line "2,500" and so
  // on. The values themselves ("0,500", "1,500" etc) are being
  // fed to the attrTween operator
  function transition() {
    linePath.transition()
      .duration(50000)
      .ease('linear')
      .attrTween("stroke-dasharray", tweenDash)
      .each("end", function() {
        // console.log(this);
        d3.select(this).call(transition);// infinite loop
      });
  } //end transition




  // this function feeds the attrTween operator above with the
  // stroke and dash lengths
  function tweenDash() {
      return function(t) {
        // console.log("linePath", linePath);
        // console.log("linePath.node()", linePath.node());
        //total length of path (single value)
        var l = linePath.node().getTotalLength();

        // this is creating a function called interpolate which takes
        // as input a single value 0-1. The function will interpolate
        // between the numbers embedded in a string. An example might
        // be interpolatString("0,500", "500,500") in which case
        // the first number would interpolate through 0-500 and the
        // second number through 500-500 (always 500). So, then
        // if you used interpolate(0.5) you would get "250, 500"
        // when input into the attrTween above this means give me
        // a line of length 250 followed by a gap of 500. Since the
        // total line length, though is only 500 to begin with this
        // essentially says give me a line of 250px followed by a gap
        // of 250px.
        interpolate = d3.interpolateString("0," + l, l + "," + l);
        //t is fraction of time 0-1 since transition began
        var marker = d3.select("#marker");

        // p is the point on the line (coordinates) at a given length
        // along the line. In this case if l=50 and we're midway through
        // the time then this would 25.
        // var p = linePath.node().getPointAtLength(t);
        var p = linePath.node().getPointAtLength(t * l);
        //Move the marker to that point
        marker.attr("transform", "translate(" + p.x + "," + p.y + ")"); //move marker
        return interpolate(t);
      }
  } //end

});
