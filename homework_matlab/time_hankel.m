function H = time_hankel(X,e_t)
% Hankle matrix algorithm for time snapshots

% e_t < nc
[nr,nc] = size(X);
H = zeros((nc-e_t+1)*nr,e_t);
for ii = 1:nr:(nc-e_t+1)*2
    % The data matrix is saved into the hankel matrix
    H(ii:ii+nr-1,:) = X(:,1:e_t);

    % The matrix X slides to the left at every iteration
    X = [X(:,2:end),zeros(nr,1)];
end

end

% old code.
% [nr,nc] = size(X);
% H = zeros(nr*nc,nc);
% for ii = 1:nr:nr*nc
%     % The data matrix is saved into the hankel matrix
%     H(ii:ii+nr-1,:) = X;
% 
%     % The matrix X slides to the left at every iteration
%     X = [X(:,2:end),zeros(nr,1)];
% end
% 
% end