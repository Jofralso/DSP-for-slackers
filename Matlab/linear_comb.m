x = [1, 2 ,3];
y = [0,-1,-2];
z = 3 * x - 2 * y;
n = 1:3;  % Generate indices for the sequences


disp('Linear combination of sequences:');
disp(z);

figure;
hold on;

stem(n, x, 'r');
stem(n, y, 'b');
stem(n, z, 'g');

xlabel('n');
ylabel('Amplitude');
title('Plot of z[n] = 3 * x - 2 * y');
hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%% Multiplicação
z2= x .* y

figure;
hold on;

stem(n, x, 'r');
stem(n, y, 'b');
stem(n, z2, 'g');

xlabel('n');
ylabel('Amplitude');
title('Plot of z2[n] = x .* y');
hold off;

%%%%%%%%%%%%%%%%%% Adicionar constante

y2 =  x + 5;


figure;
hold on;

stem(n, x, 'r');
stem(n, y2, 'b');


xlabel('n');
ylabel('Amplitude');
title('Plot of y2[n] = x + 5');
hold off;


%%%%%%%%%%%%%%%%%%%%
x = [1, 2, 3, 4];
shifted_x = [zeros(1, 2), x(1:end-2)];

n = 1:numel(x);  % Generate indices for the sequences

% Plot the sequences
figure;
stem(n, x, 'b', 'filled', 'MarkerSize', 8, 'DisplayName', 'Original x(n)');
hold on;
stem(n, shifted_x, 'r', 'filled', 'MarkerSize', 8, 'DisplayName', 'Shifted x(n)');
xlabel('n');
ylabel('Amplitude');
title('Original Sequence x(n) and Shifted Sequence x(n)');
legend;
grid on;

%%%%%%%%%%%%%%%%%%REFLECTIONS
x = [1, 2, 3, 4];
ref_x = fliplr(x);

n = 1:numel(x);

figure;
hold on;

stem(n, x, 'r');
stem(n, ref_x, 'b');


xlabel('n');
ylabel('Amplitude');
title('Reflections');
hold off;
%%%%%%%%%%%%%%%%%%% Sub Sampling

x = [1, 2, 3, 4, 5, 6];
subsamp_x = x(1:2:end); %Reter cada 2ª amostra
n = [1:6];


disp(x);
disp(subsamp_x);
figure;
hold on;

stem(n, x, 'r');
stem(n, subsamp_x, 'b');


xlabel('n');
ylabel('Amplitude');
title('Sub Sampling');
hold off;
