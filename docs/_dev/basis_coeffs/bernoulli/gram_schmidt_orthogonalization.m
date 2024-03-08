
% https://arxiv.org/pdf/2007.10814.pdf

clearvars, clc

n = 500; % up to Bn polynomial (or up to inf)


syms x real
n = n+1;
b = cell(n,1); % bernoulli
b{1} = bernoulli(0, x);
u = b; % orthogonal and normalized
c = u; % coefficients (i.e., phis) where column corresponds to x^(col-1)

% sqrt_eigval = readmatrix('BSS-ANOVA__sqrt-eigvals__500x500.txt');
c_scaled = cell(n,1);
phis = zeros(n,n);
cap = false;

for i = 2:n % i = (i-1)th B polynomial
    b{i} = bernoulli(i-1, x);
    u{i} = b{i} - int(b{i},[0,1]); % subtract u0 projection
    for j = 2:i-1 % subtract (j-1)th u projection
        u{i} = u{i} - int(b{i}*u{j},[0,1]) / int(u{j}*u{j},[0,1]) * u{j};
    end
    u{i} = simplify(u{i} / sqrt(int(u{i}*u{i},[0,1]))); % normalize

    c{i} = coeffs(u{i});
%     c_scaled{i} = c{i} * sqrt_eigval(i-1);
    c_scaled{i} = c{i};
    for j = 1:i
        phis(i,j) = c_scaled{i}(j);
        if abs(phis(i,j)) == Inf
            cap = true;
            n = i-1; % cap since at least one float64 is +/- Inf in next Bn
            break
        end
    end

    if cap
        break
    else
        phis(i,1:i)
    end
end

writematrix(phis(2:n,:), 'bernoulliPolynomials_gramSchmidt_scaled.txt')

