## Topic 3: Time-Domain and Frequency-Domain Analysis

In this topic, we'll explore the analysis of signals in both the time domain and the frequency domain. This dual perspective is crucial for understanding how signals behave and how they can be processed.

**Explanation:**

1. **Time-Domain Analysis:**
   - **Convolution:** Convolution is a mathematical operation that describes the output of a linear time-invariant system to an input signal. It's fundamental in understanding how systems respond to different inputs.
   - **Difference Equations:** These describe the relationship between the input, output, and the internal state of a discrete-time system.

2. **Frequency-Domain Analysis:**
   - **Discrete Fourier Transform (DFT):** It transforms a discrete signal from the time domain to the frequency domain, allowing us to analyze the signal's frequency components.
   - **Fast Fourier Transform (FFT):** It's an efficient algorithm to compute the DFT. Understanding FFT is crucial for real-time processing and analysis of signals.

Understanding both time and frequency domains is essential as they provide different insights into a signal. Time-domain analysis helps understand how a signal evolves over time, while frequency-domain analysis reveals its frequency components.


---
Fourier Series

Uma série de Fourier é uma expansão de uma função periódica f(x) em termos de uma soma infinita de senos e cosenos. As séries de Fourier utilizam as relações de ortogonalidade das funções seno e cosseno. O cálculo e estudo das séries de Fourier é conhecido como análise harmónica e é extremamente útil para decompor uma função periódica arbitrária numa série de termos simples que podem ser resolvidos individualmente e depois recombinados para obter a solução do problema original ou uma aproximação com a precisão desejada ou prática. Exemplos de sucessivas aproximações a funções comuns usando séries de Fourier estão ilustrados acima.

Em particular, uma vez que o princípio da sobreposição se aplica a soluções de uma equação diferencial ordinária linear homogénea, se tal equação puder ser resolvida no caso de uma única sinusóide, a solução para uma função arbitrária está imediatamente disponível expressando a função original como uma série de Fourier e depois substituindo a solução para cada componente sinusoidal. Em alguns casos especiais em que a série de Fourier pode ser somada de forma fechada, esta técnica pode até produzir soluções analíticas.

Qualquer conjunto de funções que forme um sistema ortogonal completo tem uma correspondente série de Fourier generalizada análoga à série de Fourier. Por exemplo, usando a ortogonalidade das raízes de uma função de Bessel de primeira espécie, obtemos uma chamada série de Fourier-Bessel.

O cálculo da série de Fourier (usual) baseia-se nas identidades integrais
$$
$$ \int_{-\pi}^{\pi} \sin(mx)\sin(nx)dx = \pi \delta_{mn} \qquad (1) $$
$$ \int_{-\pi}^{\pi} \cos(mx)\cos(nx)dx = \pi \delta_{mn} \qquad (2) $$
$$ \int_{-\pi}^{\pi} \sin(mx)\cos(nx)dx = 0 \qquad (3) $$
$$ \int_{-\pi}^{\pi} \sin(mx)dx = 0 \qquad (4) $$
$$ \int_{-\pi}^{\pi} \cos(mx)dx = 0 \qquad (5) $$

para \(m, n \neq 0\), onde $$\delta_{mn}$$ é o delta de Kronecker.

Usando o método para uma série de Fourier generalizada, a série de Fourier usual envolvendo senos e cossenos é obtida tomando \(f_1(x) = \cos x\) e \(f_2(x) = \sin x\). Visto que estas funções formam um sistema ortogonal completo sobre $$[- \pi, \pi]$$, a série de Fourier de uma função \(f(x)\) é dada por

$$ f(x) = \frac{1}{2}a_0 + \sum_{n=1}^{\infty} a_n\cos(nx) + \sum_{n=1}^{\infty} b_n\sin(nx), \qquad (6) $$

onde

$$ a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)dx, \qquad (7) $$
$$ a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\cos(nx)dx, \qquad (8) $$
$$ b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\sin(nx)dx, \qquad (9) $$

e \(n = 1, 2, 3, \ldots\). Note que o coeficiente do termo constante \(a_0\) foi escrito numa forma especial em comparação com a forma geral para uma série de Fourier generalizada para preservar a simetria com as definições de \(a_n\) e \(b_n\).

Os coeficientes de cosseno \(a_n\) e de seno \(b_n\) de Fourier são implementados na Linguagem Wolfram como FourierCosCoefficient[expr, t, n] e FourierSinCoefficient[expr, t, n], respectivamente.

Uma série de Fourier converge para a função \(f^_\) (igual à função original nos pontos de continuidade ou à média dos dois limites nos pontos de descontinuidade)

$$f^_ = \begin{cases}\frac{1}{2}\left[\lim_{x \to x_0^-} f(x) + \lim_{x \to x_0^+} f(x)\right], & \text{para } -\pi < x_0 < \pi; \\ \frac{1}{2}\left[\lim_{x \to -\pi^+} f(x) + \lim_{x \to \pi^-} f(x)\right], & \text{para } x_0 = -\pi, \pi. \end{cases} \qquad (10)$$

se a função satisfizer as chamadas condições de contorno de Dirichlet. O teste de Dini dá uma condição para a convergência das séries de Fourier.

FourierSeriesSquareWave

Como resultado, perto dos pontos de descontinuidade, ocorre um "anelamento" conhecido como o fenómeno de Gibbs, ilustrado acima.

Para uma função \(f(x)\) periódica num intervalo $$[-L, L]$$ em vez de $$[- \pi, \pi]$$, pode-se usar uma simples mudança de variáveis para transformar o intervalo de integração de $$[- \pi, \pi]$$ para $$[-L, L]$$. Seja

$$ x = \frac{\pi x'}{L}, \qquad (11) $$
$$ dx = \frac{\pi dx'}{L}. \qquad (12) $$

Resolvendo para $$x'$$ dá $$x' = Lx/\pi$$, e substituindo isso obtemos

$$ f(x') = \frac{1}{2}a_0 + \sum_{n=1}^{\infty} a_n\cos\left(\frac{n\pi x'}{L}\right) + \sum_{n=1}^{\infty} b_n\sin\left(\frac{n\pi x'}{L}\right). \qquad (13) $$

Portanto,

$$ a_0 = \frac{1}{L} \int_{-L}^{L} f(x')dx', \qquad  (14)$$
$$ a_n = \frac{1}{L} \int_{-L}^{L} f(x')\cos\left(\frac{n\pi x'}{L}\right)dx', \qquad (15) $$
$$ b_n = \frac{1}{L} \int_{-L}^{L} f(x')\sin\left(\frac{n\pi x'}{L}\right)dx'. \qquad (16) $$

Da mesma forma, se a função é definida no intervalo $$[0, 2L]$$, as equações acima simplesmente tornam-se

$$ a_0 = \frac{1}{L} \int_{0}^{2L} f(x')dx', \qquad (17) $$
$$ a_n = \frac{1}{L} \int_{0}^{2L} f(x')\cos\left(\frac{n\pi x'}{L}\right)dx', \qquad (18) $$
$$ b_n = \frac{1}{L} \int_{0}^{2L} f(x')\sin\left(\frac{n\pi x'}{L}\right)dx'. \qquad (19) $$

Na verdade, para $$f(x)$$ periódica com período $$2L$$, qualquer intervalo $$(x_0,x_0+2L)$$ pode ser usado, sendo a escolha uma questão de conveniência ou preferência pessoal (Arfken 1985, p. 769).

Os coeficientes para as expansões em séries de Fourier de algumas funções comuns estão dadas em Beyer (1987, pp. 411-412) e Byerly (1959, p. 51). Uma das funções mais comumente analisadas por esta técnica é a onda quadrada. As séries de Fourier para algumas funções comuns estão resumidas na tabela abaixo.

Função $$f(x)$$ | Fourier series
--- | ---
Série de Fourier - serra | $$x/(2L)$$ | $$\frac{1}{2} - \frac{1}{\pi} \sum_{n=1}^{\infty} \frac{1}{n} \sin\left(\frac{n\pi x}{L}\right)$$
Série de Fourier - onda quadrada | $$2[H(x/L) - H(x/L - 1)] - 1$$ | $$\frac{4}{\pi} \sum_{n=1,3,5,...}^{\infty} \frac{1}{n} \sin\left(\frac{n\pi x}{L}\right)$$
Série de Fourier - onda triangular | $$T(x)$$ | $$\frac{8}{\pi^2} \sum_{n=1,3,5,...}^{\infty} \frac{(-1)^{(n-1)/2}}{n^2} \sin\left(\frac{n\pi x}{L}\right)$$

Se uma função é par, de modo que $$f(x) = f(-x)$$, então $$f(x)\sin(nx)$$ é ímpar. (Isso segue porque $$\sin(nx)$$ é ímpar e um função par vezes uma função ímpar é uma função ímpar.) Portanto, $$b_n = 0$$ para todos os $$n$$. Da mesma forma, se uma função é ímpar, de modo que $$f(x) = -f(-x)$$, então $$f(x)\cos(nx)$$ é ímpar. (Isso segue porque $$\cos(nx)$$ é par e um função par vezes uma função ímpar é uma função ímpar.) Portanto, $$a_n = 0$$ para todos os $$n$$.

A noção de uma série de Fourier também pode ser estendida para coeficientes complexos. Considere uma função $$f(x)$$ real. Escreva

$$ f(x) = \sum_{n=-\infty}^{\infty} A_n e^{inx}. \qquad (20) $$

Agora examine

$$ \int_{-\pi}^{\pi} f(x)e^{-imx}dx = \int_{-\pi}^{\pi} \left(\sum_{n=-\infty}^{\infty} A_n e^{inx}\right)e^{-imx}dx $$
$$ = \sum_{n=-\infty}^{\infty} A_n \int_{-\pi}^{\pi} e^{i(n-m)x}dx $$
$$ = \sum_{n=-\infty}^{\infty} A_n \int_{-\pi}^{\pi} \left(\cos[(n-m)x] + i\sin

[(n-m)x]\right)dx $$
$$ = \sum_{n=-\infty}^{\infty} A_n 2\pi \delta_{mn} $$
$$ = 2\pi A_m, \qquad (25) $$

então

$$ A_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x)e^{-inx}dx. \qquad (26) $$

Os coeficientes podem ser expressos em termos dos da série de Fourier

$$ A_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x)[\cos(nx) - i\sin(nx)]dx $$
$$ = \begin{cases} \frac{1}{2\pi}(a_n - ib_n) & \text{para } n < 0; \\ \frac{1}{2\pi}a_0 & \text{para } n = 0; \\ \frac{1}{2\pi}(a_n + ib_n) & \text{para } n > 0. \end{cases} \qquad (29) $$

Para uma função periódica em $$[-L/2, L/2]$$, essas se tornam

$$ f(x) = \sum_{n=-\infty}^{\infty} A_n e^{i(2\pi nx/L)}, \qquad (30) $$
$$ A_n = \frac{1}{L} \int_{-L/2}^{L/2} f(x)e^{-i(2\pi nx/L)}dx. \qquad (31) $$

Essas equações são a base para a transformada de Fourier, extremamente importante, que é obtida transformando $$A_n$$ de uma variável discreta para uma contínua à medida que $$L \to \infty$$.

O coeficiente de Fourier complexo é implementado na Linguagem Wolfram como FourierCoefficient[expr, t, n].