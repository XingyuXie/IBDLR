function [U,S,V,bnd,j] = lansvdthr(varargin)

%   Lansvdthr computes singular values that are larger/smaller than the given
%   threshold svthr and also computes the corresponding singular vectors.
%
%   This code is modified from the original lansvd.m at
%   http://soi.stanford.edu/~rmunk/PROPACK/
%
%   There is not much difference from lansvd.m, except that the number of 
%   currently exactly computed singular values are computed by length(find(S > svthr)); 
%   and the rules to determine the dimension of Krylov subspace. 
%
%   Lansvdthr computes singular triplets (u,v,sigma) such that
%   A*u = sigma*v and  A'*v = sigma*u. Only a few singular values 
%   and singular vectors are computed using the Lanczos 
%   bidiagonalization algorithm with partial reorthogonalization (BPRO). 
%
%   S = LANSVD(A) 
%   S = LANSVD('Afun','Atransfun',M,N)  
%
%   The first input argument is either a  matrix or a
%   string containing the name of an M-file which applies a linear
%   operator to the columns of a given matrix.  In the latter case,
%   the second input must be the name of an M-file which applies the
%   transpose of the same operator to the columns of a given matrix,  
%   and the third and fourth arguments must be M and N, the dimensions 
%   of the problem.
%
%   [U,S,V] = LANSVD(A,svthr,'L',...) computes some largest singular
%   values above threshold svthr.
%
%   [U,S,V] = LANSVD(A,svthr,'S',...) computes some smallest singular 
%   values bellow threhold svthr.
%
%   The full calling sequence is
%
%   [U,S,V] = LANSVD(A,svthr,SIGMA,OPTIONS) 
%   [U,S,V] = LANSVD('Afun','Atransfun',M,N,svthr,SIGMA,OPTIONS)
%
%   where svthr is singular value threshold and 
%   SIGMA is 'L' or 'S'.
%
%   The OPTIONS structure specifies certain parameters in the algorithm.
%    Field name      Parameter                              Default
%   
%    OPTIONS.tol     Convergence tolerance                  16*eps
%    OPTIONS.lanmax  Dimension of the Lanczos basis.
%    OPTIONS.p0      Starting vector for the Lanczos        rand(n,1)-0.5
%                    iteration.
%    OPTIONS.delta   Level of orthogonality among the       sqrt(eps/K)
%                    Lanczos vectors.
%    OPTIONS.eta     Level of orthogonality after           10*eps^(3/4)
%                    reorthogonalization. 
%    OPTIONS.cgs     reorthogonalization method used        0
%                    '0' : iterated modified Gram-Schmidt 
%                    '1' : iterated classical Gram-Schmidt
%    OPTIONS.elr     If equal to 1 then extended local      1
%                    reorthogonalization is enforced. 
%
%   See also LANBPRO, SVDS, SVD

% References: 
% R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
%
% B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
% Prentice-Hall, Englewood Cliffs, NJ, 1980.
%
% H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
% Math. Comp. 42 (1984), no. 165, 115--142.
%
% Rasmus Munk Larsen, DAIMI, 1998
%
% Minming Chen, Algorithms and Implementations of Matrix Reconstruction, Master thesis, 
% Institute of Computing Technologies, Chinese Academy of Sciences, June 2010.
%
%%%%%%%%%%%%%%%%%%%%% Parse and check input arguments. %%%%%%%%%%%%%%%%%%%%%%

if nargin<1 | length(varargin)<1
  error('Not enough input arguments.');
end

A = varargin{1};
if ~isstr(A)    
  if ~isreal(A)
    error('A must be real')
  end  
  [m n] = size(A);
  if length(varargin) < 2, svthr=0; else  svthr=varargin{2}; end
  if length(varargin) < 3, sigma = 'L';       else  sigma=varargin{3}; end
  if length(varargin) < 4, options = [];      else  options=varargin{4}; end
else
  if length(varargin)<4
    error('Not enough input arguments.');
  end
  Atrans = varargin{2};
  if ~isstr(Atrans)
    error('Atransfunc must be the name of a function')
  end
  m = varargin{3};
  n = varargin{4};
  if length(varargin) < 5, svthr=0; else svthr=varargin{5}; end
  if length(varargin) < 6, sigma = 'L'; else sigma=varargin{6}; end  
  if length(varargin) < 7, options = []; else options=varargin{7}; end  
end

%That A being zeros or m==1 or n==1 should be specially treated.

lanmax = min(m,n);
tol = 16*eps;
p = rand(m,1)-0.5;
% Parse options struct
if isstruct(options)
  c = fieldnames(options);
  for i=1:length(c)
    if any(strcmp(c(i),'p0')), p = getfield(options,'p0'); p=p(:); end
    if any(strcmp(c(i),'tol')), tol = getfield(options,'tol'); end
    if any(strcmp(c(i),'lanmax')), lanmax = getfield(options,'lanmax'); end
  end
end

% Protect against absurd options.
tol = max(tol,eps);
lanmax = min(lanmax,min(m,n));
if size(p,1)~=m
  error('p0 must be a vector of length m')
end

lanmax = min(lanmax,min(m,n));

%%%%%%%%%%%%%%%%%%%%% Here begins the computation  %%%%%%%%%%%%%%%%%%%%%%

if strcmp(sigma,'S')
  if isstr(A) 
    error('Shift-and-invert works only when the matrix A is given explicitly.');
  else
    % Prepare for shift-and-invert Lanczos.
    if issparse(A)
      pmmd = colmmd(A);
      A.A = A(:,pmmd);
    else
      A.A = A;
    end
    if m>=n
      if issparse(A.A)
	A.R = qr(A.A,0);
	A.Rt = A.R';
	p = A.Rt\(A.A'*p); % project starting vector on span(Q1)
      else
	[A.Q,A.R] = qr(A.A,0);
	A.Rt = A.R';
	p = A.Q'*p; % project starting vector on span(Q1)
      end
    else
      error('Sorry, shift-and-invert for m<n not implemented yet!')
      A.R = qr(A.A',0);
      A.Rt = A.R';
    end
    condR = condest(A.R);    
    if condR > 1/eps
      error(['A is rank deficient or too ill-conditioned to do shift-and-' ...
	     ' invert.'])
    end
  end    
end

k = 1;
neig = 0; nrestart=-1; 
% there may be better initial j
j = round(0.05*min(m,n));
U = []; V = []; B = []; anorm = []; work = zeros(2,2);

while neig < k 

  %%%%%%%%%%%%%%%%%%%%% Compute Lanczos bidiagonalization %%%%%%%%%%%%%%%%%
  if ~isstr(A) 
    [U,B,V,p,ierr,w] = lanbpro(A,j,p,options,U,B,V,anorm);
  else
    [U,B,V,p,ierr,w] = lanbpro(A,Atrans,m,n,j,p,options,U,B,V,anorm);
  end
  work= work + w;
  
  if ierr<0 % Invariant subspace of dimension -ierr found. 
    j = -ierr;
  end

  %%%%%%%%%%%%%%%%%% Compute singular values and error bounds %%%%%%%%%%%%%%%%
  % Analyze B
  resnrm = norm(p); 
  % We might as well use the extra info. in p.
  %    S = svd(full([B;[zeros(1,j-1),resnrm]]),0); 
  %    [P,S,Q] = svd(full([B;[zeros(1,j-1),resnrm]]),0); 
  %    S = diag(S);  
  %    bot = min(abs([P(end,1:j);Q(end,1:j)]))';
		    
  [S,bot] = bdsqr(diag(B),[diag(B,-1); resnrm]);
   
% Use Largest Ritz value to estimate ||A||_2. This might save some
  % reorth. in case of restart.
  anorm=S(1);
  
  % Set simple error bounds
  bnd = resnrm*abs(bot);
  
  % Examine gap structure and refine error bounds
  bnd = refinebounds(S.^2,bnd,n*eps*anorm);
  
  % reset the estimated k
  k = length(find(S > svthr));

  %%%%%%%%%%%%%%%%%%% Check convergence criterion %%%%%%%%%%%%%%%%%%%%
  i=1;
  neig = 0;
  while i<=min(j,k) 
    if (bnd(i) <= tol*abs(S(i)))
      neig = neig + 1;
      i = i+1;
    else
      i = min(j,k)+1;
    end
  end
%   neig

  %%%%%%%%%% Check whether to stop or to extend the Krylov basis? %%%%%%%%%%
  if ierr<0 % Invariant subspace found
    if j<k
      warning(['Invariant subspace of dimension ',num2str(j-1),' found.'])
    end
    j = j-1;
    break;
  end
  if j>=lanmax % Maximal dimension of Krylov subspace reached. Bail out
    if j>=min(m,n)
      neig = k;      
      break;
    end
    if neig<k
      warning(['Maximum dimension of Krylov subspace exceeded prior',...
	    ' to convergence.']);
    end
    break;
  end
  
  % Increase dimension of Krylov subspace
  if neig>0
    % there may be better rules
    j = j + min(100,max(2,0.5*(k-neig)*j/(neig+1)));
  else
    % As long a very few singular values have converged, increase j rapidly.
    %    j = j + ceil(min(100,max(8,2^nrestart*k)));
    %     j = max(1.5*j,j+10);
    j = 2*j;
  end
%   j = ceil(min(j+1,lanmax));
  j = round(min(j,lanmax));
  nrestart = nrestart + 1;
end



%%%%%%%%%%%%%%%% Lanczos converged (or failed). Prepare output %%%%%%%%%%%%%%%

if nargout>2
  j = size(B,2);
  % Compute singular vectors
  [P,S,Q] = svd(full([B;[zeros(1,j-1),resnrm]]),0); 
  S = diag(S);
  if size(Q,2)~=k
    Q = Q(:,1:k); 
    P = P(:,1:k); 
  end
  % Compute and normalize Ritz vectors (overwrites U and V to save memory).
  if resnrm~=0
    U = U*P(1:j,:) + (p/resnrm)*P(j+1,:);
  else
    U = U*P(1:j,:);
  end
  V = V*Q;
  for i=1:k     
    nq = norm(V(:,i));
    if isfinite(nq) & nq~=0 & nq~=1
      V(:,i) = V(:,i)/nq;
    end
    nq = norm(U(:,i));
    if isfinite(nq) & nq~=0 & nq~=1
      U(:,i) = U(:,i)/nq;
    end
  end
end

% Pick out desired part the spectrum
S = S(1:k);
bnd = bnd(1:k);

if strcmp(sigma,'S')
  [S,p] = sort(-1./S);
  S = -S;
  bnd = bnd(p);
  if nargout>2
    if issparse(A.A)
      U = A.A*(A.R\U(:,p));    
      V(pmmd,:) = V(:,p);
    else
      U = A.Q(:,1:min(m,n))*U(:,p);    
      V = V(:,p);
    end
  end
end

if nargout<3
  U = S;
  S = B; % Undocumented feature -  for checking B.
else
  S = diag(S);
end
