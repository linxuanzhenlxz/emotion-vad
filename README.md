start e.g:
python avgEmoValues.py --dataPath sample_data/cn.csv --lexPath NRC-VAD-Lexicon.csv --savePath sample_data/sample_outputs/

Lexicons
The following lexicons are provided for ease of running the code in the lexicons/ folder, but must be downloaded directly from the associated sources:

The valence, arousal, and dominance lexicons

These are sourced from the NRC-VAD Lexicon.

# Mathematical Formulation of TVP-VAR Model

The **Time-Varying Parameter Vector Autoregression (TVP-VAR)** model extends the standard **VAR(p)** model by allowing its parameters to change over time. 

## 1. Standard VAR(p) Model
A standard **VAR(p)** model with \( k \) endogenous variables can be written as:

```math
y_t = c + \sum_{i=1}^{p} A_i y_{t-i} + \epsilon_t, \quad \epsilon_t \sim N(0, \Sigma)
```

where:
- \( y_t \) is a \( k \times 1 \) vector of endogenous variables.
- \( A_i \) is a \( k \times k \) coefficient matrix.
- \( c \) is a \( k \times 1 \) intercept vector.
- \( \epsilon_t \sim N(0, \Sigma) \) is the error term.

In the **TVP-VAR** model, the parameters \( c \) and \( A_i \) become time-varying:

```math
y_t = c_t + \sum_{i=1}^{p} A_{i,t} y_{t-i} + \epsilon_t, \quad \epsilon_t \sim N(0, \Sigma_t)
```

## 2. Time-Varying Parameters (TVP)
The coefficients evolve according to a random walk:

```math
\theta_t = \theta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q_t)
```

where:
- \( \theta_t \) represents the vectorized form of all coefficients.
- \( Q_t \) is the state covariance matrix.
- \( \eta_t \sim N(0, Q_t) \) is the process noise.

## 3. State-Space Representation
### Observation Equation:
```math
y_t = Z_t \theta_t + \epsilon_t, \quad \epsilon_t \sim N(0, H_t)
```

- \( Z_t \) is the **design matrix** (`self['design']`).
- \( H_t \) represents the observation noise covariance (`obs_cov`).

### State Transition Equation:
```math
\theta_t = \theta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q_t)
```

- The transition matrix is identity (`self['transition'] = np.eye(k_states)`).
- The state noise follows a Gaussian process (`state_cov`).

## 4. Stochastic Volatility Component
The model introduces **stochastic volatility**:

```math
Q_t = \text{diag}(\sigma_{1,t}^2, \sigma_{2,t}^2, ..., \sigma_{k,t}^2)
```

This is implemented as:

```python
self['state_cov'] = np.diag(np.random.uniform(0.01, 0.1, size=self.k_states))
```

## 5. Impulse Response Function (IRF) for TVP-VAR
To analyze the effect of shocks in a **time-varying environment**, we define the impulse response function (IRF) as:

```math
IRF_h = (I - \sum_{i=1}^{p} A_{i,t})^{-1} \sum_{j=0}^{h} A_{j,t}
```

where:
- \( A_{i,t} \) represents the time-varying lag coefficient matrices.
- \( h \) is the horizon of the impulse response.
- The inverse term adjusts for cumulative lagged effects.

This is implemented in the code as:

```python
lagged_matrix = lagged_coefficients.reshape(max_lag, num_vars, num_vars).sum(axis=0)
impulse_matrix = np.linalg.inv(np.eye(num_vars) - lagged_matrix)
for step in range(horizon):
    response_matrix = impulse_matrix @ np.linalg.matrix_power(lagged_matrix, step)
```

## 6. Code Mapping to Mathematical Components
| **Mathematical Concept** | **Code Implementation** |
|--------------------------|------------------------|
| VAR model | `TVPVAR(selected_data, max_lag=10)` |
| Design matrix \( Z_t \) | `self['design'] = np.zeros(...)` |
| Transition matrix \( I \) | `self['transition'] = np.eye(k_states)` |
| State noise covariance \( Q_t \) | `self['state_cov'] = np.diag(np.random.uniform(0.01, 0.1, size=self.k_states))` |
| Observation noise covariance \( H_t \) | `obs_cov=np.eye(len(selected_data.columns))` |
| Kalman filter smoothing | `sim_kfs = mod.simulation_smoother()` |
| Extracting coefficients | `dynamic_coefficients = pd.DataFrame(sim_kfs.simulated_state.T, columns=mod.state_names)` |
| Impulse Response Calculation | `compute_irf(...)` |

## 7. Lagged Variables and Interactions
The model includes **interaction terms** between `online viewer` and sentiment variables:

```math
\text{online\_Valence}_t = \text{online viewer}_t \times \text{avgValence}_t
```

Implemented as:

```python
data['online_Valence'] = data['online viewer'] * data['avgValence']
data['online_Arousal'] = data['online viewer'] * data['avgArousal']
data['online_Dominance'] = data['online viewer'] * data['avgDominance']
```

## 8. Model Estimation and Parameter Extraction
The **TVP-VAR** model is estimated using **Kalman filtering**:

```python
sim_kfs = mod.simulation_smoother()
sim_kfs.simulate()
dynamic_coefficients = pd.DataFrame(sim_kfs.simulated_state.T, columns=mod.state_names)
```

## Conclusion
This **TVP-VAR** model allows for:
- **Dynamic effects**: Sentiment and viewer interactions evolve over time.
- **Lagged dependencies**: Past audience behavior and sentiment influence current retention.
- **Stochastic volatility**: Parameters change at different speeds, capturing real-world variability.
- **Impulse Response Analysis**: The response of each variable to shocks changes dynamically over time.





