function [ ] = GenModelR( basemesh,savemesh, NLOGR, NS)

exename = 'C:\tanqingyang\splocs\splocs\ARAP.exe';
nfvmat = ['E:\nfv.mat'];
save(nfvmat,'NLOGR', 'NS');
ininame = ['E:\default.ini'];
fid = fopen(ininame,'w');
fprintf(fid,'%s\r\n',['[default]']);
fprintf(fid,'fvmat=%s\r\n',nfvmat);
fprintf(fid,'meshname=%s\r\n',basemesh);
fprintf(fid,'iternum=%d\r\n',1);
fprintf(fid,'savename=%s\r\n',savemesh);
fclose(fid);
cmdline = [exename, ' 17 ',ininame];
dos(cmdline);

end