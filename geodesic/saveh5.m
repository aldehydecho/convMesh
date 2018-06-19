function [ ] = saveh5(h5name, verts, tris)

if exist(h5name,'file')
   delete(h5name);    
end
sizev = size(verts);
sizet = size(tris);

h5create(h5name,'/verts',sizev);
h5write(h5name,'/verts',verts);
h5create(h5name,'/tris',sizet,'Datatype','int32');
h5write(h5name,'/tris',tris);

end

