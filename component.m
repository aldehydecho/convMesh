basemesh = 'scape\1.obj';
original = 'scapefeature.mat';
savepath = 'scape_convMesh50';
datapath = 'scape_convMesh50\data';

Knum = 50;

load(original);
firstr = FLOGRNEW(1,:,:);
firsts = FS(1,:,:);

[~, pointnum,~] = size(FLOGRNEW);

select = [1, 25];

rr = zeros(Knum, pointnum, 3);
rs = zeros(Knum, pointnum, 6);

rr_ = zeros(Knum, pointnum, 3);
rs_ = zeros(Knum, pointnum, 6);

l2norm = zeros(Knum, 1);

rep = zeros(Knum, 1);

for i = 1:Knum
    load([datapath, '\dimension',num2str(i),'.mat']);
    
    r = RLOGR(select,:,:)-firstr;
    s = RS(select,:,:)-firsts;
    
    diffr = sum(reshape(r,[2,pointnum*3]).^2, 2);
    diffs = sum(reshape(s,[2,pointnum*6]).^2, 2);
    diff = diffr + diffs;

    l2norm(i) = max(diff);
    
    rep(i)  = find(diff==max(diff));
    
    rr(i, :,:) = r(diff == max(diff),:,:);
    rs(i, :,:) = s(diff == max(diff),:,:);
    
    rr_(i, :,:) = r(diff == min(diff),:,:);
    rs_(i, :,:) = s(diff == min(diff),:,:);
end

[~,component_index] = sort(l2norm, 'descend');

newfolder = [savepath,'\maincomponent'];

rr = rr+firstr;
rs = rs+firsts;

rr = rr(component_index,:,:);
rs = rs(component_index,:,:);

rr_ = rr_+firstr;
rs_ = rs_+firsts;

rr_ = rr_(component_index,:,:);
rs_ = rs_(component_index,:,:);

if ~exist(newfolder,'dir')
    mkdir(newfolder);
end

copyfile( basemesh, [newfolder,'\0.obj']);

for i = 1:Knum
    reconr(:,:) = rr(i,:,:);
    recons(:,:) = rs(i,:,:);
    
    [ nr, ns ] = IFMSnew( reconr, recons );
    savemesh = [newfolder,'\',num2str(i),'_',num2str(component_index(i)),'_1.obj'];
    GenModel( basemesh,savemesh, nr, ns);
    
    
    reconr(:,:) = rr_(i,:,:);
    recons(:,:) = rs_(i,:,:);
    
    [ nr, ns ] = IFMSnew( reconr, recons );
    savemesh = [newfolder,'\',num2str(i),'_',num2str(component_index(i)),'_2.obj'];
    GenModel( basemesh,savemesh, nr, ns);
end

save('representation.mat', 'rep')