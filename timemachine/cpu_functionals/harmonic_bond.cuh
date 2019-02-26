#pragma once

template<typename NumericType>
__global__ void harmonic_bond_total_derivative(
    const NumericType *coords,
    const NumericType *params,
    const int *global_param_idxs,
    const int *param_idxs,
    const int *bond_idxs,
    NumericType *energy_out,
    NumericType *grad_out,
    NumericType *hessian_out,
    NumericType *mp_out,
    int N,
    int B) {

    const bool inference = (hessian_out == nullptr) || (mp_out == nullptr);

    auto bond_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(bond_idx < B) {
        size_t src_idx = bond_idxs[bond_idx*2+0];
        size_t dst_idx = bond_idxs[bond_idx*2+1];

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

        NumericType d3ij = dij*dij*dij;
        NumericType db = dij - b0;

        NumericType src_grad_dx = kb*db*dx/dij;
        NumericType src_grad_dy = kb*db*dy/dij;
        NumericType src_grad_dz = kb*db*dz/dij;
        NumericType dst_grad_dx = -src_grad_dx;
        NumericType dst_grad_dy = -src_grad_dy;
        NumericType dst_grad_dz = -src_grad_dz;

        atomicAdd(energy_out, kb/2.0*db*db);

        atomicAdd(grad_out + src_idx*3 + 0, src_grad_dx);
        atomicAdd(grad_out + src_idx*3 + 1, src_grad_dy);
        atomicAdd(grad_out + src_idx*3 + 2, src_grad_dz);
        atomicAdd(grad_out + dst_idx*3 + 0, dst_grad_dx);
        atomicAdd(grad_out + dst_idx*3 + 1, dst_grad_dy);
        atomicAdd(grad_out + dst_idx*3 + 2, dst_grad_dz);

        if(!inference) {

            // hessians
            // (ytz): this can be optimized pretty easily if the need ever arises.
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*dx*-dy/d3ij + dx*dy/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*dx*-dz/d3ij + dx*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*dx*dy/d3ij + dx*-dy/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*dx*dz/d3ij + dx*-dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*dy/d3ij + dx*dy/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*dy*-dz/d3ij + dy*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*dy/d3ij + -dx*dy/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*dy*dz/d3ij + dy*-dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*dz/d3ij + dx*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*dz/d3ij + dy*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*dz/d3ij + -dx*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*dz/d3ij + -dy*dz/d2ij));
            atomicAdd(hessian_out + src_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));

            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 0, kb*(db*d2x/d3ij - db/dij + -dx*dx/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 1, kb*(db*-dx*-dy/d3ij + -dx*dy/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + src_idx*3 + 2, kb*(db*-dx*-dz/d3ij + -dx*dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 0, kb*(db*-dx*dx/d3ij + db/dij + d2x/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 1, kb*(db*-dx*dy/d3ij + -dx*-dy/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 0*N*3 + dst_idx*3 + 2, kb*(db*-dx*dz/d3ij + -dx*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 0, kb*(db*-dx*-dy/d3ij + dx*-dy/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 1, kb*(db*d2y/d3ij - db/dij + -dy*dy/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + src_idx*3 + 2, kb*(db*-dy*-dz/d3ij + -dy*dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 0, kb*(db*dx*-dy/d3ij + -dx*-dy/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 1, kb*(db*-dy*dy/d3ij + db/dij + d2y/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 1*N*3 + dst_idx*3 + 2, kb*(db*-dy*dz/d3ij + -dy*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 0, kb*(db*-dx*-dz/d3ij + dx*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 1, kb*(db*-dy*-dz/d3ij + dy*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + src_idx*3 + 2, kb*(db*d2z/d3ij - db/dij + -dz*dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 0, kb*(db*dx*-dz/d3ij + -dx*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 1, kb*(db*dy*-dz/d3ij + -dy*-dz/d2ij));
            atomicAdd(hessian_out + dst_idx*3*N*3 + 2*N*3 + dst_idx*3 + 2, kb*(db*-dz*dz/d3ij + db/dij + d2z/d2ij));

            int kb_idx = global_param_idxs[param_idxs[bond_idx*2+0]];
            int b0_idx = global_param_idxs[param_idxs[bond_idx*2+1]];

            atomicAdd(mp_out + kb_idx*N*3 + src_idx*3 + 0, db*(x0 - x1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + src_idx*3 + 0, -kb*(x0 - x1)/dij );
            atomicAdd(mp_out + kb_idx*N*3 + src_idx*3 + 1, db*(y0 - y1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + src_idx*3 + 1, -kb*(y0 - y1)/dij );
            atomicAdd(mp_out + kb_idx*N*3 + src_idx*3 + 2, db*(z0 - z1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + src_idx*3 + 2, -kb*(z0 - z1)/dij );
            atomicAdd(mp_out + kb_idx*N*3 + dst_idx*3 + 0, db*(-x0 + x1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + dst_idx*3 + 0, -kb*(-x0 + x1)/dij );
            atomicAdd(mp_out + kb_idx*N*3 + dst_idx*3 + 1, db*(-y0 + y1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + dst_idx*3 + 1, -kb*(-y0 + y1)/dij );
            atomicAdd(mp_out + kb_idx*N*3 + dst_idx*3 + 2, db*(-z0 + z1)/dij );
            atomicAdd(mp_out + b0_idx*N*3 + dst_idx*3 + 2, -kb*(-z0 + z1)/dij );

            
        }


    }

}
