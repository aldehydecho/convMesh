function [ fmlogdr, fms ] = FMS( LOGDR, S ,caseid)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%LOGDR = LOGDR';
[mlogdr, nlogdr] = size(LOGDR);
[ms, ns] = size(S);
edgenum = nlogdr/9;
snum = ns/9;
%create
fmlogdr = zeros(mlogdr, edgenum,3);
fms = zeros(ms, snum,6);

for i = 1 : mlogdr
    for j = 0 : edgenum-1
        j1 = j+1;
        fmlogdr(i, j1, 1) = LOGDR(i, j*9+2);
        fmlogdr(i, j1, 2) = LOGDR(i, j*9+3);
        fmlogdr(i, j1, 3) = LOGDR(i, j*9+6);
    end
end


for i = 1 : ms
    for j = 0 : snum-1
        if nargin == 2 || caseid == 1
            j1 = j+1;
            fms(i, j1, 1) = S(i, j*9+1);
            fms(i, j1, 2) = S(i, j*9+2);
            fms(i, j1, 3) = S(i, j*9+3);
            fms(i, j1, 4) = S(i, j*9+5);
            fms(i, j1, 5) = S(i, j*9+6);
            fms(i, j1, 6) = S(i, j*9+9);
        else
            fms(i, j1, 1) = S(i, j*9+1)-1;
            fms(i, j1, 2) = S(i, j*9+5)-1;
            fms(i, j1, 3) = S(i, j*9+9)-1;
            fms(i, j1, 4) = S(i, j*9+2);
            fms(i, j1, 5) = S(i, j*9+3);
            fms(i, j1, 6) = S(i, j*9+6);
        end
    end
end

end
