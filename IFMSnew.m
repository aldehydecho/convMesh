function [ NLOGR, NS ] = IFMSnew( RLOGR, RS )
[pointnum,~] = size(RLOGR);
NLOGR = zeros(1, pointnum*9);
NS = zeros(1,pointnum*9);
for j = 0 : pointnum-1
        NLOGR(1,j*9+1) = 0;        
        NLOGR(1,j*9+2) = RLOGR(j+1, 1);%comp(1,j+1);
        NLOGR(1,j*9+3) = RLOGR(j+1, 2);%comp(2,j+1);
        NLOGR(1,j*9+4) = -RLOGR(j+1, 1);%comp(1,j+1);
        NLOGR(1,j*9+5) = 0;
        NLOGR(1,j*9+6) = RLOGR(j+1, 3);%comp(3,j+1);
        NLOGR(1,j*9+7) = -RLOGR(j+1, 2);%comp(2,j+1);
        NLOGR(1,j*9+8) = -RLOGR(j+1, 3);%comp(3,j+1);
        NLOGR(1,j*9+9) = 0;
end

for j = 0 : pointnum - 1
        NS(1, j*9+1) = RS(j+1, 1);
        NS(1, j*9+2) = RS(j+1, 2);
        NS(1, j*9+3) = RS(j+1, 3);
        NS(1, j*9+4) = RS(j+1, 2);
        NS(1, j*9+5) = RS(j+1, 4);
        NS(1, j*9+6) = RS(j+1, 5);
        NS(1, j*9+7) = RS(j+1, 3);
        NS(1, j*9+8) = RS(j+1, 5);
        NS(1, j*9+9) = RS(j+1, 6);
end

end

