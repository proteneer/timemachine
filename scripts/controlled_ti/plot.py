from jax import vmap, config, numpy as np
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from controlled_ti import pair_fxn
import matplotlib.pyplot as plt
import numpy as onp

onp.random.seed(0)

results = onp.load('results/controlled_ti.npz')
lambdas = results['lambdas']
ti_samples = results['ti_samples']
fs = results['fs']
cv_params = results['cv_params']
controlled_ti_vals = results['controlled_ti_vals']

# compute TI curves with and without CVs
ti_curve = np.mean(fs, 1)
ti_stddev = np.std(fs, 1)

controlled_ti_curve = np.mean(controlled_ti_vals, 1)
controlled_ti_stddev = np.std(controlled_ti_vals, 1)

# 1. Plot raw TI vs. controlled TI with error bands
plt.figure(figsize=(8,4))

ax = plt.subplot(1,2,1)
plt.plot(lambdas, ti_curve)
plt.fill_between(lambdas, ti_curve - ti_stddev, ti_curve + ti_stddev, alpha=0.5)
plt.title('raw TI curve\n(sample average <du/dl>)')
plt.xlabel('$\lambda$')
plt.ylabel(r'$\partial u / \partial \lambda$')

# 2. Plot controlled TI
plt.subplot(1,2,2,sharey=ax)
plt.plot(lambdas, controlled_ti_curve)
lb, ub = controlled_ti_curve - controlled_ti_stddev, controlled_ti_curve + controlled_ti_stddev
plt.xlabel('$\lambda$')
plt.ylabel(r'$\partial u / \partial \lambda - g$')
plt.fill_between(lambdas, lb, ub, alpha=0.5)
plt.title('controlled TI curve\n(sample average <du/dl - g>)')
plt.savefig('figures/ti_curves.png')
plt.close()

# 3. Plot  variance reduction
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(lambdas, ti_stddev ** 2, '.', label='raw du_dl')
plt.plot(lambdas, controlled_ti_stddev ** 2, '.', label='controlled du_dl')
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('variance')
plt.ylim(0, )
plt.title('variance profiles')

plt.subplot(2, 2, 2)
plt.plot(lambdas, (ti_stddev / controlled_ti_stddev) ** 2, '.')
plt.hlines(1, 0, 1, colors='grey', linestyles='--')
plt.xlabel('$\lambda$')
plt.ylabel('variance reduction')
#plt.yscale('log')
plt.title('variance reduction')

plt.subplot(2, 2, 3)
plt.plot(lambdas, ti_stddev ** 2, '.', label='raw du_dl')
plt.plot(lambdas, controlled_ti_stddev ** 2, '.', label='controlled du_dl')
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('variance')
plt.yscale('log')
plt.title('variance profiles\n(log scale)')
plt.ylim(1e-6)

plt.subplot(2, 2, 4)
plt.plot(lambdas, (ti_stddev / controlled_ti_stddev) ** 2, '.')
plt.hlines(1, 0, 1, colors='grey', linestyles='--')
plt.xlabel('$\lambda$')
plt.ylabel('variance reduction')
plt.yscale('log')
plt.title('variance reduction\n(log scale)')

plt.tight_layout()

plt.savefig('figures/variance_comparison.png')
plt.close()

# 4. Plot optimized control variates at all lambdas?
r = np.linspace(0, 5.0, 1000)
cmap = plt.get_cmap('viridis')
colors = cmap.colors[::len(cmap.colors) // len(lambdas)][:len(lambdas)]

plt.title('optimized control variates')
for i, lam in enumerate(lambdas):
    g_grid = vmap(pair_fxn, (0, None))(r, cv_params[i])
    plt.plot(r, g_grid, c=colors[i])
plt.xlabel('pair distance r')
plt.ylabel('test_fxn(r)')

plt.tight_layout()

plt.savefig('figures/optimized_control_variates.png')
plt.close()

plt.title('traces')
ax = plt.subplot(1,2,1)
ratios = np.nan_to_num(ti_stddev / controlled_ti_stddev)
i = np.argmax(ratios)
print(f'plotting a trace for an example where ti_stddev / controlled_ti_stddev = {ratios[i]}')
plt.plot(fs[i], '.', label='uncontrolled')
plt.title('uncontrolled')
plt.xlabel('MD snapshot')
plt.ylabel(r'$\partial u / \partial \lambda$')

plt.subplot(1,2,2,sharey=ax)
plt.plot(controlled_ti_vals[i], '.', label='controlled')
plt.xlabel('MD snapshot')
plt.ylabel(r'$\partial u / \partial \lambda - g$')
plt.title('controlled')

plt.tight_layout()

plt.savefig('figures/traces.png')
plt.close()