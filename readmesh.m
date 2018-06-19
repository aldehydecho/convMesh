function [v,f,vv] = readmesh(filename, K)
if nargin==1
    K=3;
end
[v, f, vv] = readobj(filename,K);
end



