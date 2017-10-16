
% IBDLR and RKLRR for LRR problem

clear;
close all;
addpath(genpath(cd));

fprintf('\n\n**************************************   %s   *************************************\n' , datestr(now) );


% exp 1
k = 3;    % number of subspaces
n = 100;     % samples in each subspace
d = 150;    % dimension
r = 25;      % rank

[X_cln, X,gnd] = generate_data(n,r,d,k);
[d, n]=size(X);
lambda = 3;  
gamma = 5;
%% IBDLR
tic
[Z, ~,objs] = IBDLR(X'*X,k,lambda,gamma,1);
%[Z] = RKLRR(0.21,X'*X);
time = toc;
%Z(1:N+1:end) = 0;
Z_n = cnormalize(Z, Inf);
A = abs(Z_n) + abs(Z_n)';
groups = SpectralClustering(A, 3, 'Eig_Solver', 'eigs');
imshow(A)
evalAccuracy(gnd',groups)

E = X-X*Z;
obj = lambda*nuclearnorm(Z)+sum(sqrt(sum(E.*E)));
iter = length(objs);
figure;
plot(real(objs))

fprintf('Minimum \t Time \t Iter.\n' );
fprintf('%f \t %f \t %d\n', obj, time, 0);
