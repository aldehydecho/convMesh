function []=visual(basemesh,distance,targetName)
[v,f,~]=readmesh(basemesh);
v=v';
pointCounts = size(v,1);
meshCounts = size(f,2);
rgb=imread('scale.png');
newPointCounts=pointCounts;
newOBJ = zeros(3, newPointCounts);
colorArr = zeros(3,newPointCounts);

for i = 1 : pointCounts
    P = v(i,:);
    y=int32(distance(i)*1024);
    if (y < 1)
        y=int32(1);
    end
    if (y > 1024)
        y=int32(1024);
    end
    newOBJ(:,i) = P;
    colorArr(:,i) = [rgb(y,1,1),rgb(y,1,2),rgb(y,1,3)]';
end

fid = fopen(targetName, 'w');
fprintf(fid, 'ply\n');
fprintf(fid, 'format ascii 1.0\n');
fprintf(fid, 'element vertex %d\n', newPointCounts);
fprintf(fid, 'property float x\n');
fprintf(fid, 'property float y\n');
fprintf(fid, 'property float z\n');
fprintf(fid, 'property uchar red\n');
fprintf(fid, 'property uchar green\n');
fprintf(fid, 'property uchar blue\n');
fprintf(fid, 'property uchar alpha\n');
fprintf(fid, 'element face %d\n', meshCounts);
fprintf(fid, 'property list uint8 int32 vertex_index\n');
fprintf(fid, 'end_header\n');

for i = 1 : newPointCounts
    P = newOBJ(:,i);
    color = colorArr(:,i);
    fprintf(fid, '%f %f %f %d %d %d 255\n', P(1,1),P(2,1),P(3,1),color(1,1),color(2,1),color(3,1));
    
end

for i =1 : meshCounts
    fprintf(fid, '%d %d %d %d\n', 3,f(1,i)-1,f(2,i)-1,f(3,i)-1);
end

fclose(fid);


end

