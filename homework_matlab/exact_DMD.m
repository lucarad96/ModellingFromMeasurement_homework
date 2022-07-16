function [x_dmd,eigenval] = exact_DMD(time_series,t,x_0,varargin)

% Function to compute the exact DMD (Dynamic mode Decomposition).

% INPUTS
% time_series = input data matrix, arranged in (channels X time snapshots)
% t = time vector

% OUTPUTS
% x_dmd = predicitons in time (channels X time snapshots)
% eigenvalues of the reduced "map" matrix (A_tilde)

% time interval (it should be constant!)
dt = t(2)-t(1);

% Definition of the "end-state matrix" and the "input-state matrix".
Xprime = time_series(:,2:end);
X = time_series(:,1:end-1);

% step 1 - SVD
[U,Sigma,V] = svd(X,'econ');

% Singular values to retain
if nargin>3
    if strcmp(varargin{1},'r') %("varargin","var")
        r = varargin{2};
        U = U(:,1:r);
        Sigma = Sigma(1:r,1:r);
        V = V(:,1:r);
    end
end
% Step 2 - reduced A matrix (A_tilde)
Atilde = U'*Xprime*V/Sigma;

% Step 3 - Spectral decomposition (eigenventor, eigenvalues)
[W,Lambda] = eig(Atilde);
eigenval = diag(Lambda);


% Step 4 - high-dimensional DMD modes (Phi)
omega = log(eigenval)/dt;
Phi = Xprime*(V/Sigma)*W;

% Compute mode amplitudes (b)
x_1 = U\x_0;
b = (W*Lambda)\x_1;
%b = Phi\x_0;
u_modes = b.*exp(omega*(t-t(1)));



% Forecasts
x_dmd = Phi*u_modes;

clear varargin

end