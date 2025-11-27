
# Development notes of 'get_Jacobian' method

Author: Jacob P. Krell, 2025/11/27

## Nomenclature

- $i \equiv$ index of input vector, e.g., time-index in a time series
- $j \equiv$ index of term in FoKL model, i.e., index of $\beta$ coefficient
- $y_{i} \equiv y(\mathbf{x}_{i}) \equiv y(\mathbf{x}(t_{i})) \equiv$ different ways to express the FoKL model $y$ evaluated with the input values at index $i$
- $\mathbf{J} \equiv$ Jacobian matrix of the FoKL model with respect to coefficients $\beta$
    - this is what 'get_Jacobian' returns
- $y = \beta_{0} + \beta_{1}\cdot[\dots] + \beta_{2}\cdot[\dots] + \dots \equiv$ FoKL model, where $[\dots]$ is the product of basis functions evaluated at the inputs, where the basis function orders, combinations, and input variables are defined by the interaction matrix 'mtx'

## Jacobian definition

The Jacobian is the FoKL model with respect to the beta coefficients.

$$
\mathbf{J} =
\begin{bmatrix}
\frac{d y_{0}}{d \beta_{0}} & \frac{d y_{0}}{d \beta_{1}} & \dots \\
\frac{d y_{1}}{d \beta_{0}} & \frac{d y_{1}}{d \beta_{1}} & {} \\
\vdots & {} & \ddots \\
\end{bmatrix}
$$

The Jacobian then simply becomes the interaction terms themselves, evaluated with inputs $\mathbf{x}_{i}$, without the $\beta_{j}$ coefficient.

For example, with nomenclature $x_{m}^{(i)} \equiv x_{m}(t_i)$, one might see:

- Model:
$$
y = \beta_{0} + \beta_{1} \phi_{1}(x_{1}) + \beta_{2} \phi_{1}(x_{2}) + \beta_{3} \phi_{1}(x_{1}) \phi_{1}(x_{2}) + \dots
$$

- Jacobian:
$$
\mathbf{J} =
\begin{bmatrix}
1 & \phi_{1}(x_{1}^{(0)}) & \phi_{1}(x_{2}^{(0)}) & \phi_{1}(x_{1}^{(0)}) \phi_{1}(x_{2}^{(0)}) & \dots \\
1 & \phi_{1}(x_{1}^{(1)}) & \phi_{1}(x_{2}^{(1)}) & \phi_{1}(x_{1}^{(1)}) \phi_{1}(x_{2}^{(1)}) & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots \\
\end{bmatrix}
$$

These derivatives are all exact.

## Validation testing

To validate this method, I ran the sigmoid example then called 'J = model.get_Jacobian()'.

Screenshots from this validation test are in the 'validation' subfolder.

The validation is as follows:
- Setup:
    - The sigmoid FoKL model was evaluated with 'evaluate' like normal, but with arguments such that its output should correspond to elements within the Jacobian matrix
    - Test indices of $i \in [29, 30]$ were chosen semi-arbitrarily, since both inputs $x_1$ and $x_2$ at these $i$ indices are non-zero (see _inputs_of_sigmoid.png_ at indices 28 and 29)
    - The model was evaluated with $\beta_{0} = 0$ and $\beta_{j} = 1$ for a specific term $j$, specified by row $j-1$ of mtx
    - $j = 5$ was selected because mtx[4] is a two-way interaction
- Analytic summary of setup:
    - $y = \beta_0 + \beta_j f(\mathrm{mtx})$
        - $\beta_0 = 0$ and $\beta_j = 1$ from 'betas=np.array([[0, 1]])'
        - $f(\mathrm{mtx}) = \phi_{\mathrm{mtx[0]}}(x_{1}^{(i)}) \phi_{\mathrm{mtx[1]}}(x_{2}^{(i)})$
            - 'mtx = model.mtx[4] = [1, 2]' (see _mtx_of_sigmoid.png_)
    - $\implies y_{i} = \phi_{1}(x_{1}^{(i)}) \phi_{2}(x_{2}^{(i)})$
        - $i \in [28, 29]$ from 'inputs=model.inputs[28:30]'
    - $y_{i}$ is therefore the expected value in the Jacobian at rows $i$ and column $j$
- Results:
    - _two_way_interaction_evaluated_with_beta0_at_0_and_betaj_at_1_with_sigmoid_model.PNG_ shows outputs of 0.02896057 and 0.03537156 for the two $i$ indices
    - From _Jacobian_of_sigmoid_at_test_input_1.png_ ($i=28$), 'get_Jacobian' yielded 0.02896057282289085
    - From _Jacobian_of_sigmoid_at_test_input_2.png_ ($i=29$), 'get_Jacobian' yielded 0.03537155502170189
- Conclusion:
    - The results agree, and therefore 'get_Jacobian' is validated assuming this result generalizes to the entire matrix, which there is no obvious reason to think it shouldn't

