function [ ver ] = readh5geo( feaname )
Knum = h5read(feaname,'/Gnum');
for i = 1:Knum
    compname = sprintf('/geodis%05d',i-1);
    featmp = h5read(feaname,compname);
    if  i == 1
        ver = zeros(Knum,size(featmp,1),size(featmp,2));        
    end
    ver(i,:,:) = featmp;
end

end
