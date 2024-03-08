
% Plot scaled Gram-Schmidt orthonormal Bernoulli polynomials.

clearvars, clc, close all

rows = 4;
cols = 6;
res = 500;


coeffs = readmatrix('bernoulliPolynomials_gramSchmidt_scaled_400.txt');
n = length(coeffs(:, 1));
phis = cell(n, 1);
for i = 1:n
    phis{i} = coeffs(i, 1:(i+1));
end


f = figure();
x = linspace(0, 1, res);
i = 0;
y = zeros(res, round(rows*cols));
for row = 1:rows
    for col = 1:cols
        i = i + 1;
        yi = 0;
        for j = 1:length(phis{i})
            yi = yi + phis{i}(j)*x.^(j-1);
        end
        subplot(rows, cols, i)
        plot(x, yi)
        y(:, i) = yi;
    end
end
