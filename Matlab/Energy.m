addpath('calculate_energy.m')

x1 = [1, 2, 3];
energy_x1 = calculate_energy(x1);

x2 = [1, 0.5, 0.25];
energy_x2 = calculate_energy(x2);

disp('Energy of x1:');
disp(energy_x1);
disp('Energy of x2:');
disp(energy_x2);

