Sure, I can provide you with introductory exercises on discrete time using MATLAB, Octave, and Python. These exercises will cover basic concepts such as creating sequences, plotting, and simple operations. Let's start with three exercises: one for creating a sequence, one for plotting a sequence, and one for performing basic operations on sequences.

### Exercise 1: Create a Sequence

#### MATLAB / Octave:
```matlab
% MATLAB / Octave
n = 0:10;  % Generate a sequence from 0 to 10
x = 2.^n;  % Compute the sequence x(n) = 2^n
disp('Sequence x(n) = 2^n:')
disp(x);
```

#### Python:
```python
# Python
import numpy as np

n = np.arange(0, 11)  # Generate a sequence from 0 to 10
x = 2 ** n  # Compute the sequence x(n) = 2^n
print('Sequence x(n) = 2^n:')
print(x)
```

### Exercise 2: Plot a Sequence

#### MATLAB / Octave:
```matlab
% MATLAB / Octave
stem(n, x);  % Plot the sequence using stem plot
xlabel('n');
ylabel('x[n]');
title('Plot of x[n] = 2^n');
```

#### Python:
```python
# Python
import matplotlib.pyplot as plt

plt.stem(n, x)  # Plot the sequence using stem plot
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Plot of x[n] = 2^n')
plt.show()
```

### Exercise 3: Basic Operations on Sequences

#### MATLAB / Octave:
```matlab
% MATLAB / Octave
n = 0:10;  % Generate a sequence from 0 to 10
x = sin(n);  % Compute the sequence x(n) = sin(n)

y = 0.5 * x;  % Scale the sequence by a factor of 0.5
disp('Scaled sequence y(n) = 0.5 * x(n):');
disp(y);

z = x + y;  % Add the sequences
disp('Sum of sequences z(n) = x(n) + y(n):');
disp(z);
```

#### Python:
```python
# Python
n = np.arange(0, 11)  # Generate a sequence from 0 to 10
x = np.sin(n)  # Compute the sequence x(n) = sin(n)

y = 0.5 * x  # Scale the sequence by a factor of 0.5
print('Scaled sequence y(n) = 0.5 * x(n):')
print(y)

z = x + y  # Add the sequences
print('Sum of sequences z(n) = x(n) + y(n):')
print(z)
```

### Exercise 4: Linear Combination

#### Exercise:
Given two sequences \(x[n] = \{1, 2, 3\}\) and \(y[n] = \{0, -1, -2\}\), calculate the linear combination \(z[n] = 3x[n] - 2y[n]\).

#### MATLAB / Octave:
```matlab
x = [1, 2, 3];
y = [0, -1, -2];
z = 3 * x - 2 * y;  % Linear combination
disp('Linear combination of sequences:');
disp(z);
```

#### Python:
```python
x = [1, 2, 3]
y = [0, -1, -2]
z = 3 * np.array(x) - 2 * np.array(y)  # Linear combination
print('Linear combination of sequences:')
print(z)
```

### Exercise 5: Sequence Multiplication

#### Exercise:
Given two sequences \(x[n] = \{1, 2, 3\}\) and \(y[n] = \{0, -1, -2\}\), calculate the point-wise multiplication \(z[n] = x[n] \cdot y[n]\).

#### MATLAB / Octave:
```matlab
x = [1, 2, 3];
y = [0, -1, -2];
z = x .* y;  % Point-wise multiplication
disp('Point-wise multiplication of sequences:');
disp(z);
```

#### Python:
```python
x = [1, 2, 3]
y = [0, -1, -2]
z = np.multiply(x, y)  # Point-wise multiplication
print('Point-wise multiplication of sequences:')
print(z)
```

### Exercise 6: Adding a Constant

#### Exercise:
Add a constant value of 5 to the sequence \(x[n] = \{1, 2, 3\}\).

#### MATLAB / Octave:
```matlab
x = [1, 2, 3];
y = x + 5;  % Adding a constant
disp('Sequence after adding a constant:');
disp(y);
```

#### Python:
```python
x = [1, 2, 3]
y = np.add(x, 5)  # Adding a constant
print('Sequence after adding a constant:')
print(y)
```

### Exercise 7: Shifting

#### Exercise:
Shift the sequence \(x[n] = \{1, 2, 3, 4\}\) to the right by 2 positions.

#### MATLAB / Octave:
```matlab
x = [1, 2, 3, 4];
shifted_x = circshift(x, 2);  % Circular shift to the right by 2 positions
disp('Sequence after shifting:');
disp(shifted_x);
```

#### Python:
```python
x = [1, 2, 3, 4]
shifted_x = np.roll(x, 2)  # Circular shift to the right by 2 positions
print('Sequence after shifting:')
print(shifted_x)
```

### Exercise 8: Reflection

#### Exercise:
Reflect the sequence \(x[n] = \{1, 2, 3\}\) around the \(n=0\) axis.

#### MATLAB / Octave:
```matlab
x = [1, 2, 3];
reflected_x = fliplr(x);  % Reflect the sequence
disp('Sequence after reflection:');
disp(reflected_x);
```

#### Python:
```python
x = [1, 2, 3]
reflected_x = np.flip(x)  # Reflect the sequence
print('Sequence after reflection:')
print(reflected_x)
```

### Exercise 9: Subsampling

#### Exercise:
Subsample the sequence \(x[n] = \{1, 2, 3, 4, 5, 6\}\) by keeping every second sample.

#### MATLAB / Octave:
```matlab
x = [1, 2, 3, 4, 5, 6];
subsampled_x = x(1:2:end);  % Retain every second sample
disp('Subsampled sequence:');
disp(subsampled_x);
```

#### Python:
```python
x = [1, 2, 3, 4, 5, 6]
subsampled_x = x[::2]  # Retain every second sample
print('Subsampled sequence:')
print(subsampled_x)
```

### Exercise 10: Oversampling

#### Exercise:
Perform oversampling on the sequence \(x[n] = \{1, 2, 3\}\) by inserting zero samples between the original samples.

#### MATLAB / Octave:
```matlab
x = [1, 2, 3];
oversampled_x = zeros(1, 2*numel(x)-1);  % Initialize with zeros
oversampled_x(1:2:end) = x;  % Insert original samples
disp('Oversampled sequence:');
disp(oversampled_x);
```

#### Python:
```python
x = [1, 2, 3]
oversampled_x = np.zeros(2 * len(x) - 1)  # Initialize with zeros
oversampled_x[::2] = x  # Insert original samples
print('Oversampled sequence:')
print(oversampled_x)
```

### 1. **Unit Impulse (\(\delta[n]\)) and Unit Step (\(u[n]\)) Sequences:**

#### Unit Impulse (\(\delta[n]\)):
- The unit impulse sequence \(\delta[n]\) is defined as:
  - \(\delta[n] = 1\) for \(n = 0\)
  - \(\delta[n] = 0\) for \(n \neq 0\)
  
**Explanation:**
- The unit impulse represents an infinitesimally short signal with an area of 1 centered at \(n = 0\).

#### MATLAB / Octave Example:
```matlab
n = -5:5;
delta = (n == 0);  % Create unit impulse sequence
stem(n, delta);
xlabel('n');
ylabel('\delta[n]');
title('Unit Impulse Sequence');
```

#### Python Example:
```python
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-5, 6)
delta = np.where(n == 0, 1, 0)  # Create unit impulse sequence
plt.stem(n, delta, use_line_collection=True)
plt.xlabel('n')
plt.ylabel('$\delta[n]$')
plt.title('Unit Impulse Sequence')
plt.show()
```

### 2. **Exponential Sequence (\(x[n] = \alpha^n\)):**

- Exponential sequences have the form \(x[n] = \alpha^n\) where \(\alpha\) is a constant.

**Explanation:**
- The value of \(\alpha\) determines how the sequence behaves over time. For \(0 < \alpha < 1\), the values decrease; for \(\alpha > 1\), they increase.

#### MATLAB / Octave Example:
```matlab
n = 0:10;
alpha1 = 0.5;
alpha2 = 2;
x1 = alpha1 .^ n;
x2 = alpha2 .^ n;
stem(n, x1, 'r', 'DisplayName', '0.5^n');
hold on;
stem(n, x2, 'b', 'DisplayName', '2^n');
legend;
xlabel('n');
ylabel('x[n]');
title('Exponential Sequences');
```

#### Python Example:
```python
n = np.arange(0, 11)
alpha1 = 0.5
alpha2 = 2
x1 = alpha1 ** n
x2 = alpha2 ** n

plt.stem(n, x1, 'r', label='$0.5^n$')
plt.stem(n, x2, 'b', label='$2^n$')
plt.legend()
plt.xlabel('n')
plt.ylabel('$x[n]$')
plt.title('Exponential Sequences')
plt.show()
```

### 3. **Sinusoidal Sequence (\(x[n] = A \cdot \cos(\omega_0 n + \phi)\)):**

- A sinusoidal sequence is defined by \(x[n] = A \cdot \cos(\omega_0 n + \phi)\) where \(A\) is the amplitude, \(\omega_0\) is the angular frequency, and \(\phi\) is the phase angle.

**Explanation:**
- The amplitude \(A\) determines the peak value of the waveform.
- The angular frequency \(\omega_0\) controls the rate of oscillation.
- The phase angle \(\phi\) determines the horizontal shift of the waveform.

#### MATLAB / Octave Example:
```matlab
n = 0:40;
A = 1.5;  % Amplitude
omega_0 = pi/8;  % Angular frequency
phi = pi/3;  % Phase angle

x = A * cos(omega_0 * n + phi);  % Sinusoidal sequence
stem(n, x);
xlabel('n');
ylabel('x[n]');
title('Sinusoidal Sequence');
```

#### Python Example:
```python
n = np.arange(0, 41)
A = 1.5  # Amplitude
omega_0 = np.pi/8  # Angular frequency
phi = np.pi/3  # Phase angle

x = A * np.cos(omega_0 * n + phi)  # Sinusoidal sequence
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('$x[n]$')
plt.title('Sinusoidal Sequence')
plt.show()
```

### 4. **Exponential Complex Sequence (\(x[n] = \alpha^n \cdot e^{j\omega_0 n}\)):**

- Exponential complex sequences combine an exponential term with a sinusoidal term affected by the exponential.

**Explanation:**
- The exponential term \(\alpha^n\) controls the growth or decay of the oscillations.
- The sinusoidal term \(e^{j\omega_0 n}\) introduces the oscillatory behavior.
- The magnitude of \(\alpha\) determines whether oscillations decrease, increase, or stay constant.

#### MATLAB / Octave Example:
```matlab
n = 0:20;
alpha = 0.8;
omega_0 = pi/8;

x = alpha .^ n .* exp(1j * omega_0 * n);  % Exponential complex sequence
stem(n, abs(x));  % Plot magnitude
xlabel('n');
ylabel('|x[n]|');
title('Exponential Complex Sequence');
```

#### Python Example:
```python
n = np.arange(0, 21)
alpha = 0.8
omega_0 = np.pi/8

x = alpha ** n * np.exp(1j * omega_0 * n)  # Exponential complex sequence
plt.stem(n, np.abs(x))  # Plot magnitude
plt.xlabel('n')
plt.ylabel('|x[n]|')
plt.title('Exponential Complex Sequence')
plt.show()
```

### 5. **Duration: Finite vs. Infinite:**

- **Finite Duration**:
  - Sequences that have values only within a finite range of indices (\(n\)).
  - \(x[n]\) has non-zero values only within a specific range of \(n\).

- **Infinite Duration**:
  - Sequences that have values across an infinite range of indices (\(n\)).
  - There is no specific range where \(x[n]\) is zero for all \(n\).

**Explanation:**
- Finite duration sequences have a limited range of non-zero values, making them useful in practical applications with defined time intervals.
- Infinite duration sequences have non-zero values across an infinite range, often representing idealized or theoretical cases.

#### Finite Duration Example:
```matlab
n = -5:5;
x_finite = (n >= -3) & (n <= 3);  % Finite duration sequence
stem(n, x_finite);
xlabel('n');
ylabel('x[n]');
title('Finite Duration Sequence');
```

#### Infinite Duration Example:
```matlab
n = -10:10;
x_infinite = sin(n);  % Infinite duration (sinusoidal) sequence
stem(n, x_infinite);
xlabel('n');
ylabel('x[n]');
title('Infinite Duration Sequence (Sinusoidal)');
```

### 6. **Causality: Causal vs. Non-Causal:**

- **Causal**:
  - Sequences for which \(x[n] = 0\) for \(n < 0\).
  - Information in the sequence starts at or after \(n = 0\).

- **Non-Causal**:
  - Sequences for which \(x[n] \neq 0\) for \(n < 0\).
  - Information in the sequence may start before \(n = 0\).

**Explanation:**
- Causal sequences are often associated with real-world processes where the effect follows the cause in time.
- Non-causal sequences may involve theoretical or hypothetical scenarios.

#### Causal Example:
```matlab
n = 0:10;
x_causal = (n >= 0);  % Causal sequence
stem(n, x_causal);
xlabel('n');
ylabel('x[n]');
title('Causal Sequence');
```

#### Non-Causal Example:
```matlab
n = -5:5;
x_noncausal = ones(1, length(n));  % Non-causal sequence
stem(n, x_noncausal);
xlabel('n');
ylabel('x[n]');
title('Non-Causal Sequence');
```

### 7. **Symmetric and Antisymmetric Sequences:**

- **Symmetric Conjugate**:
  - For a real sequence \(x[n]\), if \(x[n] = x^*(-n)\), it's symmetric.
  - If \(x[n]\) is real, \(x[n] = x(-n)\) (even function).

- **Antisymmetric Conjugate**:
  - For a real sequence \(x[n]\), if \(x[n] = -x^*(-n)\), it's antisymmetric.
  - If \(x[n]\) is real, \(x[n] = -x(-n)\) (odd function).

**Explanation:**
- Symmetric sequences have even symmetry around the y-axis, while antisymmetric sequences have odd symmetry.

#### MATLAB / Octave Example (Symmetric):
```matlab
n = -5:5;
x_symmetric = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1];  % Symmetric sequence
stem(n, x_symmetric);
xlabel('n');
ylabel('x[n]');
title('Symmetric Sequence');
```

#### MATLAB / Octave Example (Antisymmetric):
```matlab
n = -5:5;
x_antisymmetric = [0, -1, -2, -3, -4, 4, 3, 2, 1, 0];  % Antisymmetric sequence
stem(n, x_antisymmetric);
xlabel('n');
ylabel('x[n]');
title('Antisymmetric Sequence');
```

### 8. **Periodic and Aperiodic Sequences:**

- **Periodic Sequences**:
  - \(x[n]\) is periodic if \(x[n] = x[n + N]\) for some positive integer \(N\).
  - The smallest \(N\) is called the fundamental period.

- **Aperiodic Sequences**:
  - \(x[n]\) is aperiodic if it's not periodic.

**Explanation:**
- Periodic sequences repeat their pattern after a certain number of samples (period), whereas aperiodic sequences do not.

#### MATLAB / Octave Example (Periodic):
```matlab
n = 0:15;
x_periodic = sin(2*pi/8 * n);  % Periodic sinusoidal sequence
stem(n, x_periodic);
xlabel('n');
ylabel('x[n]');
title('Periodic Sinusoidal Sequence');
```

#### MATLAB / Octave Example (Aperiodic):
```matlab
n = -5:5;
x_aperiodic = exp(n);  % Aperiodic exponential sequence
stem(n, x_aperiodic);
xlabel('n');
ylabel('x[n]');
title('Aperiodic Exponential Sequence');
```

### 9. **Periodic Sinusoidal Sequence Properties:**

- **Properties of a Periodic Sinusoidal Sequence**:
  - A sinusoidal sequence \(x[n] = A \cos(\omega_0 n + \phi)\) is periodic if and only if:
    - \(\omega_0\) is a rational multiple of \(2\pi\), i.e., \(\omega_0 = \frac{2\pi k}{N}\) where \(k\) is an integer and \(N\) is the period.

**Explanation:**
- The frequency \(\omega_0\) must be a rational multiple of \(2\pi\) for the sequence to be periodic.

#### MATLAB / Octave Example:
```matlab
n = 0:39;
A = 2;  % Amplitude
k = 3;  % Integer
N = 10;  % Period

omega_0 = 2*pi*k/N;
x_periodic = A * cos(omega_0 * n);  % Periodic sinusoidal sequence
stem(n, x_periodic);
xlabel('n');
ylabel('x[n]');
title('Periodic Sinusoidal Sequence');
```

### 10. **Exponential Complex Sequences and Oscillation Behavior:**

- **Oscillation Behavior in Exponential Complex Sequences**:
  - The magnitude of \(|\alpha|\) in \(x[n] = \alpha^n \cdot e^{j\omega_0 n}\) affects the oscillations:
    - \(|\alpha| < 1\): Oscillations decrease in amplitude.
    - \(|\alpha| > 1\): Oscillations increase in amplitude.
    - \(|\alpha| = 1\): No change in amplitude (pure oscillations).

**Explanation:**
- The magnitude of \(\alpha\) affects how the oscillations in the sequence behave: decrease, increase, or remain constant in amplitude.

#### MATLAB / Octave Example:
```matlab
n = 0:30;
alpha_values = [0.5, 1.5, -0.8, -1.5];
x = zeros(length(alpha_values), length(n));

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    x(i, :) = alpha .^ n .* exp(1j * 0.2 * pi * n);
end

figure;
for i = 1:length(alpha_values)
    subplot(length(alpha_values), 1, i);
    stem(n, abs(x(i, :)));
    xlabel('n');
    ylabel('|x[n]|');
    title(['|x[n]| for \alpha = ', num2str(alpha_values(i))]);
end
```

### Explanation:
- We generate four sequences with different \(\alpha\) values to demonstrate how the magnitude of \(\alpha\) influences the oscillation behavior of the exponential complex sequence.

### 11. **Durations: Finite, Infinite, and Bi-lateral Sequences:**

- **Finite Duration Sequences**:
  - \(x[n]\) has non-zero values only within a specific range of \(n\).
  - The sequence has a limited range of non-zero values.

- **Infinite Duration Sequences (Non-Finite)**:
  - \(x[n]\) is non-zero for an infinite number of values of \(n\).

- **Bi-lateral Sequences**:
  - Sequences where \(x[n]\) is non-zero for both positive and negative \(n\).

**Explanation:**
- Finite duration sequences are limited to a specific range of \(n\), while infinite duration sequences have non-zero values for an infinite number of \(n\).
- Bi-lateral sequences have non-zero values for both positive and negative \(n\).

#### MATLAB / Octave Example (Finite Duration):
```matlab
n1 = -5:5;
x_finite = (n1 >= -3) & (n1 <= 3);  % Finite duration sequence
stem(n1, x_finite);
xlabel('n');
ylabel('x[n]');
title('Finite Duration Sequence');
```

#### MATLAB / Octave Example (Infinite Duration):
```matlab
n2 = -10:10;
x_infinite = sin(n2);  % Infinite duration (sinusoidal) sequence
stem(n2, x_infinite);
xlabel('n');
ylabel('x[n]');
title('Infinite Duration Sequence (Sinusoidal)');
```

#### MATLAB / Octave Example (Bi-lateral):
```matlab
n3 = -5:5;
x_bilateral = [0, -1, -2, -3, -4, 4, 3, 2, 1, 0];  % Bi-lateral sequence
stem(n3, x_bilateral);
xlabel('n');
ylabel('x[n]');
title('Bi-lateral Sequence');
```

### 12. **Causality: Causal and Non-Causal Sequences:**

- **Causal Sequences**:
  - \(x[n] = 0\) for \(n < 0\).
  - Information in the sequence starts at or after \(n = 0\).

- **Non-Causal Sequences**:
  - \(x[n] \neq 0\) for \(n < 0\).
  - Information in the sequence may start before \(n = 0\).

**Explanation:**
- Causal sequences have their non-zero values only for \(n \geq 0\), typically representing real-world processes where the effect follows the cause in time.
- Non-causal sequences have non-zero values for \(n < 0\), which may not follow a cause-and-effect relationship.

#### MATLAB / Octave Example (Causal):
```matlab
n1 = 0:10;
x_causal = (n1 >= 0);  % Causal sequence
stem(n1, x_causal);
xlabel('n');
ylabel('x[n]');
title('Causal Sequence');
```

#### MATLAB / Octave Example (Non-Causal):
```matlab
n2 = -5:5;
x_noncausal = ones(1, length(n2));  % Non-causal sequence
stem(n2, x_noncausal);
xlabel('n');
ylabel('x[n]');
title('Non-Causal Sequence');
```

### 13. **Periodic and Aperiodic Sequences:**

- **Periodic Sequences**:
  - \(x[n]\) is periodic if \(x[n] = x[n + N]\) for some positive integer \(N\).
  - The smallest \(N\) is called the fundamental period.

- **Aperiodic Sequences**:
  - \(x[n]\) is aperiodic if it's not periodic.

**Explanation:**
- Periodic sequences repeat their pattern after a certain number of samples (period), whereas aperiodic sequences do not.

#### MATLAB / Octave Example (Periodic):
```matlab
n1 = 0:15;
x_periodic = sin(2*pi/8 * n1);  % Periodic sinusoidal sequence
stem(n1, x_periodic);
xlabel('n');
ylabel('x[n]');
title('Periodic Sinusoidal Sequence');
```

#### MATLAB / Octave Example (Aperiodic):
```matlab
n2 = -5:5;
x_aperiodic = exp(n2);  % Aperiodic exponential sequence
stem(n2, x_aperiodic);
xlabel('n');
ylabel('x[n]');
title('Aperiodic Exponential Sequence');
```

### 14. **Periodic Sinusoidal Sequences Properties:**

- **Properties of a Periodic Sinusoidal Sequence**:
  - A sinusoidal sequence \(x[n] = A \cos(\omega_0 n + \phi)\) is periodic if and only if:
    - \(\omega_0\) is a rational multiple of \(2\pi\), i.e., \(\omega_0 = \frac{2\pi k}{N}\) where \(k\) is an integer and \(N\) is the period.

**Explanation:**
- The frequency \(\omega_0\) must be a rational multiple of \(2\pi\) for the sequence to be periodic.

#### MATLAB / Octave Example:
```matlab
n = 0:39;
A = 2;  % Amplitude
k = 3;  % Integer
N = 10;  % Period

omega_0 = 2*pi*k/N;
x_periodic = A * cos(omega_0 * n);  % Periodic sinusoidal sequence
stem(n, x_periodic);
xlabel('n');
ylabel('x[n]');
title('Periodic Sinusoidal Sequence');
```
### 15. **Symmetric and Antisymmetric Sequences:**

- **Symmetric Conjugate**:
  - For a real sequence \(x[n]\), if \(x[n] = x^*(-n)\), it's symmetric.
  - If \(x[n]\) is real, \(x[n] = x(-n)\) (even function).

- **Antisymmetric Conjugate**:
  - For a real sequence \(x[n]\), if \(x[n] = -x^*(-n)\), it's antisymmetric.
  - If \(x[n]\) is real, \(x[n] = -x(-n)\) (odd function).

**Explanation:**
- Symmetric sequences have even symmetry around the y-axis, while antisymmetric sequences have odd symmetry.

#### MATLAB / Octave Example (Symmetric):
```matlab
n = -5:5;
x_symmetric = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1];  % Symmetric sequence
stem(n, x_symmetric);
xlabel('n');
ylabel('x[n]');
title('Symmetric Sequence');
```

#### MATLAB / Octave Example (Antisymmetric):
```matlab
n = -5:5;
x_antisymmetric = [0, -1, -2, -3, -4, 4, 3, 2, 1, 0];  % Antisymmetric sequence
stem(n, x_antisymmetric);
xlabel('n');
ylabel('x[n]');
title('Antisymmetric Sequence');
```

### 16. **Oversampling:**

- **Oversampling**:
  - Inserting zero samples between the original samples in a sequence.

**Explanation:**
- Oversampling increases the density of samples in the sequence by adding zero samples, which can be useful in certain signal processing applications.

#### MATLAB / Octave Example:
```matlab
x = [1, 2, 3];
oversampled_x = zeros(1, 2*numel(x)-1);  % Initialize with zeros
oversampled_x(1:2:end) = x;  % Insert original samples
disp('Original sequence:');
disp(x);
disp('Oversampled sequence:');
disp(oversampled_x);
```

### 17. **Decimation (Subsampling):**

- **Decimation (Subsampling)**:
  - Retaining only every \(M\)th sample and discarding the rest.

**Explanation:**
- Decimation is commonly used in signal processing to reduce the data rate and focus on specific components of a signal.

#### MATLAB / Octave Example:
```matlab
x = 1:20;
M = 2;  % Decimation factor
decimated_x = x(1:M:end);  % Retain every Mth sample
disp('Original sequence:');
disp(x);
disp('Decimated sequence:');
disp(decimated_x);
```


### 18. **Energy | Power:**

#### Fundamental Concepts - Energy and Power:

- **Energy**:
  - Energy in a discrete-time signal \(x[n]\) over a finite range is given by:
  \[ E = \sum_{n} |x[n]|^2 \]
  - Energy is finite for sequences with finite amplitude and duration.

- **Power**:
  - Power in a discrete-time signal \(x[n]\) is given by the average energy per sample:
  \[ P = \lim_{{N \to \infty}} \frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2 \]
  - Power is finite for sequences with non-zero values over an infinite range.

#### When \(x[n]\) is periodic:

- **Energy Sequences**: Finite Energy and Zero Power
  - Examples: Finite sequences or sequences with decaying amplitude.

- **Power Sequences**: Infinite Energy and Finite Power
  - Examples: Periodic sequences.

### MATLAB / Octave Examples:

#### Energy of a Sequence:
```matlab
function energy = calculate_energy(x)
    energy = sum(abs(x).^2);
end

x1 = [1, 2, 3];
energy_x1 = calculate_energy(x1);

x2 = [1, 0.5, 0.25];
energy_x2 = calculate_energy(x2);

disp('Energy of x1:');
disp(energy_x1);
disp('Energy of x2:');
disp(energy_x2);
```

#### Power of a Sequence:
```matlab
function power = calculate_power(x)
    N = length(x);
    power = sum(abs(x).^2) / N;
end

x3 = [1, 2, 3, 4, 5];  % Finite duration sequence
power_x3 = calculate_power(x3);

x4 = sin(2*pi/10 * (0:99));  % Periodic sequence
power_x4 = calculate_power(x4);

disp('Power of x3:');
disp(power_x3);
disp('Power of x4:');
disp(power_x4);
```

### 1. **Energy | Power:**

#### Fundamental Concepts - Energy and Power:

- **Energy**:
  - Energy in a signal \(x[n]\) measures its total magnitude over a finite range of samples.
  - **Formula**: \(E = \sum_{n} |x[n]|^2\)
  - Think of energy as the "total strength" or "total content" of the signal.

- **Power**:
  - Power in a signal \(x[n]\) measures its average strength per sample.
  - **Formula**: \(P = \lim_{{N \to \infty}} \frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2\)
  - Think of power as the "average strength" of the signal.

#### When \(x[n]\) is periodic:

- **Energy Sequences**:
  - **Characteristics**: Finite Energy, Zero Power.
  - **Examples**: Sequences with decaying amplitude or finite duration.

- **Power Sequences**:
  - **Characteristics**: Infinite Energy, Finite Power.
  - **Examples**: Periodic sequences.

### Simpler Examples:

- **Energy Example**:
  - For a sequence \(x[n] = [1, 2, 3]\),
  - **Energy**: \(E = |1|^2 + |2|^2 + |3|^2 = 14\)

- **Power Example**:
  - For a sequence \(x[n] = [1, 2, 3, 4, 5]\),
  - **Power**: \(P = \frac{1}{5}(|1|^2 + |2|^2 + |3|^2 + |4|^2 + |5|^2)\)

### MATLAB / Octave Examples:

#### Energy Calculation:
```matlab
x1 = [1, 2, 3];
energy_x1 = sum(abs(x1).^2);

x2 = [1, 0.5, 0.25];
energy_x2 = sum(abs(x2).^2);

disp('Energy of x1:');
disp(energy_x1);
disp('Energy of x2:');
disp(energy_x2);
```

#### Power Calculation:
```matlab
x3 = [1, 2, 3, 4, 5];
power_x3 = mean(abs(x3).^2);

x4 = sin(2*pi/10 * (0:99));
power_x4 = mean(abs(x4).^2);

disp('Power of x3:');
disp(power_x3);
disp('Power of x4:');
disp(power_x4);
```
