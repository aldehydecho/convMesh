basemesh = 'scape\1.obj';
original = 'scapefeature.mat';

load(original);
firstr(:,:) = FLOGRNEW(1,:,:);
firsts(:,:) = FS(1,:,:);

% recondata = 'scape_convMesh50\data\recover.mat';
recondata = 'scape_convMesh50\data\synthesis.mat';

load(recondata);

% savepath = 'scape_convMesh50\reconstruct';
savepath = 'scape_convMesh50\synthesis';

if ~exist(savepath,'dir')
    mkdir(savepath);
end

[modelnum, pointnum,~] = size(RLOGR);

for i = 1:modelnum
    r(:,:) = RLOGR(i,:,:);
    s(:,:) = RS(i,:,:);
    
    [ nr, ns ] = IFMSnew( r, s );
    savemesh = [savepath,'\',num2str(i),'.obj'];
    GenModel( basemesh, savemesh, nr, ns);
    
    % the following code is used to demonstrate the strength of the
    % deformation, not reconstruction error (as explained in Sec. 5)
    % -- start --
    pointdiff = sum(cat(2, (r-firstr).^2,(s-firsts).^2), 2);
    pointdiff = sqrt(pointdiff);
    pointdiff = pointdiff/max(pointdiff);
    savefile = [savepath,'\',num2str(i),'.ply'];
    visual(savemesh, pointdiff, savefile);
    % -- end --

end