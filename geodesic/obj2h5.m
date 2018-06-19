function [] = obj2h5(objname, h5name)
[v, f, ~] = readmesh( objname );
f = f-1;
tris = int32(f);
saveh5(h5name, v, tris);
end