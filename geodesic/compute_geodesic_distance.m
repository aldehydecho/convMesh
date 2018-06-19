function [point_geodesic]=compute_geodesic_distance(input)
[ex, ~, ~]=fileparts(input);
if ~exist([ex,'\_tem_'],'dir')
    mkdir([ex,'\_tem_'])
end

inh5=[ex,'\_tem_\input.h5'];
obj2h5(input,inh5);
outh5=[ex,'\_tem_\output.h5'];

pycmd=['python.exe test.py ',inh5,' ',outh5];
system(pycmd);
point_geodesic=readh5geo(outh5);
[~, ~, ~]=rmdir([ex,'\_tem_'],'s');

end