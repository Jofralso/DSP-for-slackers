n = 0:10
x = sin(n)
y=0.5 * x;
disp('Sequence x(n) = sin(n)');
disp(x)
disp('Scaled sequence y(n) = 0.5* x(n):');
disp(y);

z = x +y;
disp('Sum of sequences z(n)=x(n)+y(n)');
disp(z);

%Plot the sequences
figure;
stem(n, x, 'b', 'filled', 'MarkerSize', 8, 'DisplayName', 'x[n]=sin[n]');
hold on;
stem(n, y, 'r', 'filled', 'MarkerSize', 8, 'DisplayName', 'y[n] = 0.5* x[n]');
stem(n, z, 'g', 'filled', 'MarkerSize', 8, 'DisplayName', 'z[n]= x[n]+  y[n]');
xlabel('n');
ylabel('Amplitude');
legend;
grid on;


