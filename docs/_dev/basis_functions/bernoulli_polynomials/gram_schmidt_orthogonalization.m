n = 200; % highest order of Bernoulli polynomial (Bn)
rows = 4; % rows in plot
cols = 6; % columns in plot

% --------------------------------------------------------------------------------------------
% Gram-Schmidt:

syms x real
n = n + 1;
b = cell(n, 1); % Bn
b{1} = bernoulli(0, x);
u = b; % orthogonal and normalized Bn
c = u; % coefficients (i.e., phis) where column corresponds to x^(col-1)
phis = zeros(n, n); % coefficients as matrix (not cell)
cap = false; % true when a coefficient is +/-Inf

for i = 2:n % (i-1)th Bn
    b{i} = bernoulli(i-1, x);
    u{i} = b{i} - int(b{i}, [0,1]); % subtract u0 projection
    for j = 2:i-1 % subtract (j-1)th u projection
        u{i} = u{i} - int(b{i}*u{j}, [0,1]) / int(u{j}*u{j}, [0,1]) * u{j};
    end
    u{i} = simplify(u{i} / sqrt(int(u{i}*u{i}, [0,1]))); % normalize

    c{i} = coeffs(u{i});
    for j = 1:i
        phis(i,j) = c{i}(j);
        if abs(phis(i,j)) == Inf
            cap = true;
            n = i-1; % previous order is max without +/-Inf
            break
        end
    end

    if cap
        break
    end
  
end

writematrix(phis(2:n, 1:n), 'orthogonal_Bn_normalized.txt') % note '1:n' should be same as ':'

% --------------------------------------------------------------------------------------------
% Plot:

f = figure();
i = 0;
for row = 1:rows
    for col = 1:cols
        i = i + 1;
        subplot(rows, cols, i)
        fplot(u{i+1}, [0, 1])
    end
end

saveas(f, 'orthogonal_Bn_normalized__fplot.png')

