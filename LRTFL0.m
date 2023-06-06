function [A,B,C,X]=LRTFL0(Y,A,B,C,R,lambda,alpha,tau)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: Y: noisy tensor with I,J, K size
%        A: Initialization for A with the size of I, L*R
%        B: Initialization for B with the size of J, L*R
%        C: Initialization for C with the size of K, R
%        R: the dimentionality of subspace: estimated by Hysime
%        lambda: parameter of nuclear norm: default parm: 3
%        alpha: parameter of L0 gradient regularization, defalut: 1/(sqrt(I*J))
%        tau: parameter of sparse term , defalut: 300/(sqrt(I*J))
% Output: X resulted tensor
%         A, B, C: factor matrices
% Reference:F. Xiong, J. Zhou and Y. Qian, "Hyperspectral Restoration via L? Gradient Regularized Low-Rank Tensor Factorization," in IEEE Transactions on Geoscience and Remote Sensing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



tic
% Y=(Y);
sizeD           = size(Y);
normD           = norm(Y(:));
n               = prod(sizeD);
maxIter         = 50;
epsilon         = 1e-6;
out_value       = [];
out_value.SSIM  = [];
out_value.PSNR  = [];
out_value.ERGAS = [];
%%  Initialization
X               = Y;      % X : The clean image
Z               = X;                % Z : auxiliary variable for X
S               = ((zeros(sizeD)));     % S : sparse Yse
F               = ((zeros(3*n,1)));     % F : auxiliary variable for tv
Gamma           = F;                % The multiplier for DZ-F
M1              = ((zeros(size(Y)))); % The multiplier for
M2              = M1;
h               = sizeD(1);
w               = sizeD(2);
d               = sizeD(3);
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;
mu1=0.01;
for iter=1:maxIter
    preX       = X;
    %% - update A and B, C and X
    G       = 1/2*(Y-S+Z+(M1-M2)/mu1);
   [A,B,C,X]=btdNuclear(G,A,B,C,R,lambda/mu1);
  
    % - update Z
    z          = Z(:);
    z          = myPCG1(z,X,M2,F,Gamma,mu1,sizeD);
    Z          = reshape(z,sizeD);
    diff_Z     = diff3(Z(:), sizeD);
    argZ=diff_Z+ Gamma/mu1;
    tempx=reshape(argZ(1:n,1),sizeD);
    tempy=reshape(argZ(1+n:2*n,1),sizeD);
    tempz=reshape(argZ(1+2*n:3*n,1),sizeD);
    t=(tempx.^2+tempy.^2+tempz.^2)<=alpha/mu1;
    tempx(t)=0;
    tempy(t)=0;
    tempz(t)=0;
    F=[tempx(:);tempy(:);tempz(:)];
    %% - update S
    S          = softthre(Y-X+M1/mu1,tau/mu1);% sparse
    %     S          = mu1*(Y-X+M1/mu1)/(mu1+2*lambda);
    %% - update M
    M1         = M1 + mu1*(Y-X-S);
    M2         = M2 + mu1*(X-Z);
    %% - update Gamma
    Gamma      = Gamma+mu1*(diff_Z-F);
 mu1       = min(mu1 *1.5,1e6);
    %% compute the error
    
    errList    = norm(X(:)-preX(:)) / normD;
    fprintf('LRTFL0: iterations = %d   difference=%f\n', iter, errList);
    if errList < epsilon
        break;
    end
end
end



function [A,B,C,X]=btdNuclear(G,A,B,C,R,mu2)
[I,J,K]=size(G);
G1=tens2mat(G,1);
G2=tens2mat(G,2);
G3=tens2mat(G,3);
times1=0;
iter2=1;
L=min(I,J);
for r=1:R
    C(:,r)= C(:,r)./norm( C(:,r),'fro');
end
dom=norm(G(:),'fro');
M=myKr2(A,B,ones(1,R)*L,ones(1,R)*L);
X=reshape(M*C',I,J,K);

while 1
    oldX=X;
    M=myKr(C,B,ones(1,R),ones(1,R)*L);
    A=(G1*M)/(M'*M+mu2*eye(size(M,2)));
    M=myKr(C,A,ones(1,R),ones(1,R)*L);
    B=(G2*M)/(M'*M+mu2*eye(size(M,2)));
    M=myKr2(A,B,ones(1,R)*L,ones(1,R)*L);
    for r=1:R
        mr        =    M(:,r);
        temp         = G3-C*M'+C(:,r)*mr';
        cr        =    temp*mr;
        cr        =    cr./norm(cr,2);
        C(:,r)    =    cr;
    end
    
    X=reshape(M*C',I,J,K);
    
    iter2=iter2+1;
    tol1=norm(oldX(:)-X(:),'fro')/dom;
    if(tol1<1e-4||iter2>100)
        times1=times1+1;
    else
        times1=0;
    end
    
    if(times1==1)
        break;
    end
    if mod(iter2,1)==0
        fprintf(strcat('In Nuclear BTD:\t Iter \t',num2str(iter2),'\t',num2str(tol1),'\n'));
    end
end
end

