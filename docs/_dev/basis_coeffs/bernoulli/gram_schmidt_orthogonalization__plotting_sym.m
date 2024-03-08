
% Symbolically plot Gram-Schmidt orthonormal Bernoulli polynomials.

% clearvars, clc, close all
% load('i_358_is_done.mat')

clc, close all
rows = 4;
cols = 6;


f = figure();
i = 0;
for row = 1:rows
    for col = 1:cols
        i = i + 1;
        subplot(rows, cols, i)
        fplot(u{i+1}, [0, 1])
    end
end

