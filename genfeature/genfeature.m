function genfeature(modelfolder, featurename)
cmd=['genfeature.exe ',modelfolder];
dos(cmd);
load([modelfolder,'\FeatureMat1.mat']);
delete([modelfolder,'\FeatureMat1.mat'])
[FLOGRNEW, FS] = FMS(LOGRNEW, S);
save([modelfolder,'\',featurename,'.mat'],'FLOGRNEW','FS','-v7.3');
end