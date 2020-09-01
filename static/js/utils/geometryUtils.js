/**
* @file geometryUtils.js Managing geometrical questions 
* as how many and which points are inside a polygon
* @author Leslie Solorzano
* @see {@link geometryUtils}
*/
geometryUtils={}

/** 
* @param {Bool} a 
* @param {Bool} b
* XOR between two booleans 
*/
geometryUtils.XOR=function(a,b) {
	return ( a || b ) && !( a && b );
};


/** 
* @param {Number} x X global image coordinate 
* @param {Number} y Y global image coordinate
* @param {Object} region TM json region
* Fast algorithm to find if a global coordinate is inside a polygon (convex or not)
* adapted from this code http://alienryderflex.com/polygon/
*/
geometryUtils.globalPointInRegion= function(x,y,region) {
	var polyCorners=region.len;
	var i, j=polyCorners-1; 
	var oddNodes=false;
	var points=region.globalPoints;
	
	for (i=0; i<polyCorners; i++) {
		if ((points[i].y< y && points[j].y>=y || points[j].y< y && points[i].y>=y) &&  (points[i].x<=x || points[j].x<=x)) {
			oddNodes =geometryUtils.XOR(oddNodes, (points[i].x+(y-points[i].y)/(points[j].y-points[i].y)*(points[j].x-points[i].x)<x) ) 
		}
		j=i; 
	}
	
	return oddNodes; 
};

/** 
* @param {Number} x X width-normalized coordinate 
* @param {Number} y Y width-normalized coordinate
* @param {Object} region TM json region
* Fast algorithm to find if a point is inside any polygon (convex or not)
* adapted from this code http://alienryderflex.com/polygon/
*/
geometryUtils.viewerPointInRegion= function(x,y,region) {
	var polyCorners=region.len;
	var i, j=polyCorners-1; 
	var oddNodes=false;
	var points=region.points;
	
	for (i=0; i<polyCorners; i++) {
		if ((points[i].y< y && points[j].y>=y || points[j].y< y && points[i].y>=y) &&  (points[i].x<=x || points[j].x<=x)) {
			oddNodes =geometryUtils.XOR(oddNodes, (points[i].x+(y-points[i].y)/(points[j].y-points[i].y)*(points[j].x-points[i].x)<x) ) 
		}
		j=i; 
	}
	
	return oddNodes; 
}
