#include "integrator.cuh"
#include <ctime>
#include <iostream>
#include <vector>
#include <stdexcept>

void test_hvp() {

    timemachine::Integrator<float> intg;
    
    // const int N3 = 2500*3;
    // const int P = 500;

    const int N3 = 5;
    const int P = 3;

    float *A, *B, *C;
    A = (float*)malloc(N3*N3*sizeof(float));
    B = (float*)malloc(P*N3*sizeof(float));
    C = (float*)malloc(P*N3*sizeof(float));
 
    // Allocate device storage for A,B,C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N3*N3*sizeof(float));
    cudaMalloc((void**)&d_B, P*N3*sizeof(float));
    cudaMalloc((void**)&d_C, P*N3*sizeof(float));
 


    // Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    // Matrices are arranged column major
    for(size_t i=0; i<N3; i++) {
        for(size_t j=i; j<N3; j++) {

            if(i==j) {
                A[i*N3 + j] = sin(i*N3+j);
            } else {
                A[i*N3 + j] = cos(i*N3+j);
                A[j*N3 + i] = cos(i*N3+j);
            }
        }   
    }

    for(size_t i=0; i < N3; i++) {
        for(size_t j=0; j < N3; j++) {
            std::cout << A[i*N3+j] << " " ;
        }
        std::cout << std::endl;
    }
 
    for(size_t p=0; p<P; p++) {
        for(size_t j=0; j<N3; j++) {
            size_t index = p*N3 + j;
            if(p==j) {
                B[index] = sin(index);
                C[index] = cos(index)*cos(index);
            }
        else {
                B[index] = sin(index)*sin(index);
                C[index] = sin(index)*cos(index);;
            }
        }   
    }

    cublasErrchk(cublasSetMatrix(N3, N3, sizeof(float), A, N3, d_A, N3));
    cublasErrchk(cublasSetMatrix(N3, P, sizeof(float), B, N3, d_B, N3));
    cublasErrchk(cublasSetMatrix(N3, P, sizeof(float), C, N3, d_C, N3));
 
    // reference implementation
    std::vector<float> expected(P*N3, 0);
    for(int p=0; p < P; p++) {
        for(int row=0; row < N3; row++) {
            float sum = 0;
            for(int col=0; col < N3; col++) {
                sum += A[row*N3 + col] * B[p*N3 + col];
            }
            expected[p*N3 + row] += sum + C[p*N3 + row];
        }
    }

    std::clock_t start;
    float duration;
    start = std::clock();
    intg.hessian_vector_product(d_A, d_B, d_C, N3, P);

    cudaDeviceSynchronize();
    duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    std::cout <<"speed: "<< duration/10 <<'\n';

    cublasErrchk(cublasGetMatrix(N3, P, sizeof(float), d_C, N3, C, N3));

    for(int p=0; p < P; p++) {
        for(int row=0; row < N3; row++) {
            float ref = expected[p*N3 + row];
            float test = C[p*N3+row];
            if(fabs(ref - test) > 1e-3) {
                std::cout << ref << " " << test << std::endl;
                throw(std::runtime_error("FATAL"));
            }
        }
    }
}

void test_reduce() {

    const int W = 100;
    const int P = 500;
    const int N3 = 2500*3;

    // const int W = 10;
    // const int P = 5;
    // const int N3 = 12;



    float *WPN3 = (float*)malloc(W*P*N3*sizeof(float));
 
    for(size_t w=0; w < W; w++) {
        for(size_t p=0; p < P; p++) {
            for(size_t i=0; i<N3; i++) {
                WPN3[w*P*N3 + p*N3 + i] = sin(w+p+i);
            }
        }   
    }



    float *coeff_bs = (float*)malloc(P*N3*sizeof(float));
    for(size_t p=0; p < P; p++) {
        for(size_t i=0; i<N3; i++) {
            coeff_bs[p*N3 + i] = cos(p+i);
        }
    }

    // Allocate device storage for A,B,C
    float *d_WPN3;
    float *d_coeff_bs;

    gpuErrchk(cudaMalloc((void**)&d_WPN3, W*P*N3*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_coeff_bs, P*N3*sizeof(float)));

// __host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) 

    gpuErrchk(cudaMemcpy(d_WPN3, WPN3, W*P*N3*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeff_bs, coeff_bs, P*N3*sizeof(float), cudaMemcpyHostToDevice));

    size_t tpb = 32;
    const size_t tot = P*N3;
    size_t n_blocks = (tot + tpb - 1) / tpb;

    float coeff_a = 0.9;

    int starting_k = 0;


    std::clock_t start;
    float duration;
    start = std::clock();

    // for(size_t i=0; i < 10; i++ ) {
        // std::cout << n_blocks << " " << tpb << std::endl;
        reduce_total<float><<<n_blocks, tpb>>>(
            coeff_a,
            d_coeff_bs,
            d_WPN3,
            starting_k,
            W,
            P*N3
        );
    // }

    gpuErrchk(cudaDeviceSynchronize());



    duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    std::cout <<"speed: "<< duration <<'\n';

    std::vector<float> test(P*N3);

    gpuErrchk(cudaMemcpy(&test[0], d_WPN3, P*N3*sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected(P*N3);
    for(size_t p=0; p < P; p++) {
        for(size_t i=0; i<N3; i++) {
            float accum = 0;
            float prefactor = 0.0;
            float a_n = 1.0;
            for(size_t w=0; w < W; w++) {
                prefactor += a_n;
                a_n *= coeff_a;
                accum += prefactor*WPN3[w*P*N3 + p*N3 + i];
            }
            expected[p*N3 + i] = coeff_bs[p*N3 + i]*accum;
        }
    }

    for(size_t p=0; p < P; p++) {
        for(size_t i=0; i < N3; i++) {
            float a = test[p*N3+i];
            float b = expected[p*N3+i];
            if(fabs(a-b) > 1e-4) {
                throw(std::runtime_error("lol"));
            }

        }
    }

}

int main() {


    test_reduce();

}