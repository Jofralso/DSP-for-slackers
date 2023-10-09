n = 0:30;
alpha_values = [0.5, 1.5, -0.8, -1.5];
x = zeros(length(alpha_values), length(n));

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    x(i, :) = alpha .^ n .* exp(1j * 0.2 * pi * n);
    
    subplot(length(alpha_values), 1, i);
    stem(n, abs(x(i, :)));
    xlabel('n');
    ylabel('|x[n]|');
    title(['|x[n]| for \alpha = ', num2str(alpha_values(i))]);
end