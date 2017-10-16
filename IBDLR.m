function [Z, J, obj]= KLRBD(K,k,lambda,gamma,display)
%
% Written by Xingyu Xie (nuaaxing@gmail.com), March 2017.
%
%clear global;
%global A K temp B M2 L_const beta ZminJ;
%global temp B ;
if nargin < 5
    display = false;
end
[~, n] = size(K);

maxiter = 1000;
normK = norm(K,2);
mu0 = normK*0.1;
mu = mu0;
min_mu = 1e-8;
min_beta = 1e-9;
beta = normK*0.1;
rho_beta = 0.95;
rho_mu = 0.95;

tau = normK*1.1; % Lipschitz constant
tol2 = 1e-5;
stopD_ZJ = tol2;
tol1 = 1e-4;
I_1 = ones(n,1);
%Z = Z0;
Z0 = zeros(n,n);
Z = Z0;
J = Z;
W2 = 1.0./sqrt((diag(K)+mu^2));    
KZM = Z0;
KM = K*diag(W2);
opt.tol = 1e-6;%precision for computing the partial SVD
opt.p0 = I_1;
%down_normfK = 1/norm(K,'fro');
if display
   obj = zeros(maxiter,1);
end

% %the initial guess of the rank of Z is 5.
A.U = zeros(n,3);%the left singluar vectors of Z
A.s = zeros(3,1);%the singular values of Z
A.V = zeros(n,3);%the right singular vectors of Z
%sv = 5;
for t = 1 : maxiter
    
   % calculate Z: 
   ZminJ = (Z-J);
   L_const = mu/(beta*tau+mu);
   temp = Z - L_const*(beta*(KZM-KM)+ZminJ);
   thres = (beta*lambda*L_const);
   [U,sigma,V] = lansvdthr(temp, thres, 'L', opt);
   %[U,sigma,V] = svd(temp,'econ');
   sigma = diag(sigma);
   svp = length(find(sigma>thres));
    if svp>=1
        sigma = sigma(1:svp)-thres;
    else
        svp = 1;
        sigma = 0;
        U = Z0;
        V = Z0;
    end
     A.U= U(:,1:svp);A.V = V(:,1:svp); A.Sigm= diag(sigma);A.s = sigma;
     Z = A.U*A.Sigm*A.V';
   
   %Update W
   B = (abs(J) + abs(J')).*0.5;
   %[W,~,~] = laneig(B,k,'AS',opt);
   [W,~] = eig(B);
   len = min(length(W(1,:)),k);
   W = W(:,len)*W(:,len)';
   %W = W(:,k)*W(:,k)';
   %Update J
   tmp_W = diag(W)*I_1'-W;
   tmp = abs(Z)-0.5*beta*gamma*(tmp_W+tmp_W');
   J = max(0,tmp);
   J = sign(Z).*J;
   % update mu
   mu = max(min_mu,mu*rho_mu);
   beta = max(min_beta, beta*rho_beta);
   
   
   if (t==2 || mod(t,20)==0)
       %stop_ZJ = sqrt(abs(norm(A.s)^2 + norm(A_old.s)^2 - 2*sum(sum((A.Sigm*(A.V'*A_old.V)).*((A.U'*A_old.U)*A_old.Sigm)))))*down_normfK;
       stop_ZJ = max(max(abs(Z-Z_old)));
       %stop_ZJ = max(norm(Z_old-Z,'fro')*down_normfK, norm(J_old-J,'fro')*down_normfK);
       stopD_ZJ_old = stopD_ZJ;
       %stopD_ZJ =sqrt(abs(norm(A.s)^2+sum(sum(J.*J))-2*sum(sum((J*A.V).*(A.U*A.Sigm)))))*down_normfK;
       stopD_ZJ = max(max(abs(Z-J)));
       
       if stop_ZJ<tol2  &&  (stopD_ZJ<tol1 || abs(stopD_ZJ_old - stopD_ZJ)<1e-9)
            if display && (t==2 || mod(t,20)==0)
                disp(['iter ' num2str(t) ',mu=' num2str(mu,'%2.1e') ...
                    ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',difference=' num2str(stop_ZJ,'%2.3e') ',Z-J=' num2str(stopD_ZJ ,'%2.3e')]);
            end
            break;
       end 
   end
   % compute the objective function value
   if display
       E_tmp = Z'*K*Z+K-K*Z-Z'*K;
       M_tmp = sqrt(diag(E_tmp)); 
       obj(t) = lambda*nuclearnorm(Z)+sum(M_tmp)+gamma*trace(W'*(diag(B*I_1)-B));
   end 
   if display && (t==2 || mod(t,20)==0)
       disp(['iter ' num2str(t) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',difference=' num2str(stop_ZJ,'%2.3e') ',Z-J=' num2str(stopD_ZJ ,'%2.3e')]);
   end
   

   Z_old = Z;

   % calculate M which is a diagonal matrix
   KU = K*A.U;
   KZ = KU*A.Sigm*A.V';
   ZKZ =  A.V*A.Sigm*(A.U'*KU)*A.Sigm*A.V';
   E_F = ZKZ+K-KZ-KZ';
   M2 = 1.0./sqrt(diag(E_F)+mu^2);   
   M = repmat(M2',n,1);
   KM = K.*M;
   KZM = KZ.*M;

end
if display
    if t<maxiter
       obj(t:end) = []; 
    end   
else    
    obj = [];
end



