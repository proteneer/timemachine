## `charges`
* `bootstrap_am1.py` -- fit a SMIRKS-based model to approximate AM1BCCELF10 charges from `oequacpac`

## `noneq_adaptation`
* `adapt_noneq.py` -- Adaptively choose lambda spacing for a nonequilibrium switch, by monitoring a population of "in-progress" switching trajectories. Here, nonequilibrium switches are implemented by alternating "propagation" ("running MD at a fixed value of lambda") and "perturbation" ("incrementing lambda"). Works are computed numerically as a sum of `u(x, lam[t+1]) - u(x, lam[t])` increments.
* `deploy.py` -- Run nonequilibrium MD using `ctxt.multiple_steps(lambda_schedule)` with `lambda_schedule` given by interpolating the discrete schedule found in `adapt_noneq.py`. (Instead of making a big increment to "lambda" every hundred MD steps or so, make a small lambda increment every step.) Works are computed numerically by `trapz(du_dl, lambda_schedule)`.