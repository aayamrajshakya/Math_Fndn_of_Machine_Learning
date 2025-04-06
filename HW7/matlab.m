A = randn(8,4);
A(:,5:6) = A(:,1:2)+A(:,3:4);
[Q,R] = qr(randn(6));
A = A*Q;

% 7.5 a) Print out A
disp('Matrix A:')
disp(A);

% 7.5 b) Obtain singular values of A
singular_val = svd(A);
format short e;
disp('Singular values of A (scientific notation):')
disp(singular_val);

% 7.5 c) Confirm numerical rank
num_rank = rank(A);
disp("Numerical rank of A is:")
disp(num_rank);

% 7.5 d) Rank with threshold
help rank
%% syntax --> k = rank(A, tol)
%%% threshold of 1e-16 (smallest exponent among all 6 singular values)
rank_w_threshold = rank(A, 1e-16);
disp("New rank:")
disp(rank_w_threshold);