template<typename NumericType>
void harmonic_bond_grad(
    const int num_atoms,
    const int num_params,
    const NumericType *coords,
    const NumericType *params,
    const int num_bonds,
    const int *bond_idxs,
    const int *param_idxs,
    NumericType *grads) {
    // size_t num_bonds = bond_idxs.size()/2;
    for(int bond_idx=0; bond_idx < num_bonds; bond_idx++) {
        int src_idx = bond_idxs[bond_idx*2 + 0];
        int dst_idx = bond_idxs[bond_idx*2 + 1];

        NumericType x0 = coords[src_idx*3+0];
        NumericType y0 = coords[src_idx*3+1];
        NumericType z0 = coords[src_idx*3+2];
        NumericType x1 = coords[dst_idx*3+0];
        NumericType y1 = coords[dst_idx*3+1];
        NumericType z1 = coords[dst_idx*3+2];

        NumericType dx = x0 - x1;
        NumericType dy = y0 - y1;
        NumericType dz = z0 - z1;

        NumericType kb = params[param_idxs[bond_idx*2+0]];
        NumericType b0 = params[param_idxs[bond_idx*2+1]];

        NumericType d2x = dx*dx;
        NumericType d2y = dy*dy;
        NumericType d2z = dz*dz;

        NumericType d2ij = d2x + d2y + d2z;
        NumericType dij = sqrt(d2ij);

        NumericType db = dij - b0;

        NumericType src_grad_dx = kb*db*dx/dij;
        NumericType src_grad_dy = kb*db*dy/dij;
        NumericType src_grad_dz = kb*db*dz/dij;
        NumericType dst_grad_dx = -src_grad_dx;
        NumericType dst_grad_dy = -src_grad_dy;
        NumericType dst_grad_dz = -src_grad_dz;

        // energy += kb/2.0*db*db;

        grads[src_idx*3 + 0] += src_grad_dx;
        grads[src_idx*3 + 1] += src_grad_dy;
        grads[src_idx*3 + 2] += src_grad_dz;
        grads[dst_idx*3 + 0] += dst_grad_dx;
        grads[dst_idx*3 + 1] += dst_grad_dy;
        grads[dst_idx*3 + 2] += dst_grad_dz;
    }

}
