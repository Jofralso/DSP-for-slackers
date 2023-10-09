n = 0:10;  % Generate a sequence from 0 to 10
x = 2.^n;  % Compute the sequence x(n) = 2^n
disp('Sequence x(n) = 2^n:')
disp(x);
stem(n, x);  % Plot the sequence using stem plot
xlabel('n');
ylabel('x[n]');
title('Plot of x[n] = 2^n');
