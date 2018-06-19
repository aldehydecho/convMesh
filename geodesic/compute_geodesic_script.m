basemesh = '..\scape\1.obj';

[point_geodesic]=compute_geodesic_distance(basemesh);

dmin = 0.2;
dmax = 0.4;

distance = (point_geodesic-dmin)/(dmax-dmin);
distance(distance>1) = 1;
distance(distance<0) = 0;

save('..\scapedistance.mat', 'distance');