basemesh = 'scape\1.obj';

[~,~,VV] = readmesh(basemesh,3);

[pointnum,~] = size(VV);

maxdegree = 0;

degrees = zeros(pointnum, 1);

for i = 1:pointnum
    degree = length(VV{i});
    degrees(i) = degree;
    if degree > maxdegree
        maxdegree = degree;
    end
end

neighbour = zeros(pointnum, maxdegree);

for i = 1:pointnum
    neighbour(i,1:degrees(i)) = VV{i};
end

save('scapeneighbour.mat', 'neighbour');