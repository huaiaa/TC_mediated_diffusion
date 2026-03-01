#include<iostream>
#include <fstream>
#include <cufft.h> 
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cassert>
/////double
//#define Double cufftDoubleReal  
//#define CufftForward CUFFT_D2Z
//#define CufftBackward CUFFT_Z2D
//#define CufftComplex cufftDoubleComplex
//#define CufftExcForward cufftExecD2Z
//#define CufftExcBackward cufftExecZ2D
// const int batch = 4; const int cross_total = 14*3+9; const int chain_total = 29+22+20+10;
//const int max_cross_per = 14; const int max_chain_per = 29;
//const int batch_cross_num[batch] = { 14,14,14,9 };
//const int batch_chain_num[batch] = { 29,22,20,10 };
////////float

//#define ISDOUBLE
#ifdef ISDOUBLE
const bool IsDouble = true;
#define Double cufftDoubleReal  
#define CufftForward CUFFT_D2Z
#define CufftBackward CUFFT_Z2D
#define CufftComplex cufftDoubleComplex
#define CufftExcForward cufftExecD2Z
#define CufftExcBackward cufftExecZ2D
#else
const bool IsDouble = false;
#define Double cufftReal  
#define CufftForward CUFFT_R2C
#define CufftBackward CUFFT_C2R
#define CufftComplex cufftComplex
#define CufftExcForward cufftExecR2C
#define CufftExcBackward cufftExecC2R
#endif


#define threads 512
#define negative_error -0.1
#define FLT_MIN 1.175494351e-38F
//const int batch = 1; const int cross_total = 27; const int chain_total = 81;
//const int max_chain_per = 6; const int max_cross_per = max_chain_per* 2;
//const int batch_cross_num[batch] = { 27 };
//const int batch_chain_num[batch] = { 81 };
//////////
using namespace std;
const int BOX_N = 80;
//const int block_num = 3;
//const int block_type[3] = {0,1,0};
const int CROSSLINK_NUM = 64;
const int CHAIN_NUM = 192;
#define NANO_M 20
const Double RECEPTOR_NUM = NANO_M;
//const int F_num = 7;
//#define rA_id 0
//#define rB_id 1
//#define rAbb_id 2
//#define rAbf_id 3

#define pi 3.1415926535

#define Axis3to1(i,j,k) i*N*N+j*N+k
//#define e1 1.0
//#define e2 -1.0
//Double eps_cpu[6] = { 1.0,1.0, 1.0,1.0,-1.0,1.0 };//AA,AB,BB,AAb,BAb,AbAb
//#define eps_AA 0
//#define eps_AB 1
//#define eps_BB 2
//#define eps_AAb 3
//#define eps_BAb 4
//#define eps_AbAb 5

const int VOLUME = BOX_N * BOX_N * BOX_N;
//unsigned int size_DOUBLE = V * sizeof(Double);
//unsigned int size_COMPLEX = V * sizeof(CufftComplex);
//unsigned int size_INT = V * sizeof(int);
#define DR 0.1
#define DR3 0.001

//const Double DR = 0.1;
//const Double DR3 = DR * DR * DR;
const Double DS = 0.02;
const Double CHAIN_LENGTH = 4.0;
const int ETA_NUM = 200;
const Double S0 = 2.0;
const int ETA_NUM_S0 = 100;
const int BATCH_SIZE = 1;
const Double V0 = 1.0;

/////batch info/////

//const int batch = 1; const int cross_total =81*2; const int chain_total =81;
//const int max_cross_per = 2; const int max_chain_per = 1;
//const int batch_cross_num[batch] = { 2 };
////const int batch_cross_num[batch] = { 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };
////const int batch_chain_num[batch] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
//const int batch_chain_num[batch] = { 1 };
/////*********/////
//const int threads_per_block = 512;
//const int number_of_blocks = (V + threads_per_block - 1) / threads_per_block;
//const int Vm = V * max_cross_per;
//const int number_of_blocks_n = (Vm + threads_per_block - 1) / threads_per_block;
//const int number_of_blocks_sum = (V*eta_num + threads_per_block - 1) / threads_per_block;
/////////////Debug
void Cuda_err_print(int signal) {
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\nSIGNAL:%d\n", cudaGetErrorString(err), signal);
        exit(-1);
    }
}
/////////////
void String_append(char* s, char* s1, char* s2) {
    sprintf(s, s1, s2);
}
void String_append(char* s, char* s1, int s2) {
    sprintf(s, s1, s2);
}
void SaveData(Double* data, const char* filename, int length) {
    FILE* output = fopen(filename, "wb");
    fwrite(data, sizeof(Double), length, output);
    fclose(output);
}
void SaveData(int* data, const char* filename, int length) {
    FILE* output = fopen(filename, "wb");
    fwrite(data, sizeof(int), length, output);
    fclose(output);
}
bool SaveComplex(CufftComplex* data, const char* filename1, int length) {
    FILE* output = fopen(filename1, "wb");

    if (output == NULL) {
        printf("无法打开文件");
        return false;
    }
    fwrite(data, sizeof(CufftComplex), length, output);
    fclose(output);
    return true;
}
Double log_fact(int N)
{
    Double result = 0.0;
    for (Double i = 1.0; i <= N + 0.1; i += 1.0)
    {
        result += log(i);
    }
    return result;
}
//////
__global__ void Sum2D_gpu(Double* p, int axis, int shape0, int shape1, bool IsOdd, int l, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int j, k;
        if (axis == 0) {
            j = i / shape1;
            k = i % shape1;
            if (j == 0 && IsOdd) {

            }
            else {
                p[i] += p[(2 * l - 1 - j) * shape1 + k];
            }
        }
        else {
            j = i / shape0;
            k = i % shape0;
            if (j == 0 && IsOdd) {

            }
            else {
                p[k * shape1 + j] += p[k * shape1 + (2 * l - 1 - j)];
            }
        }
    }
}
__global__ void Positive_ensure(Double* a, int* error, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        if (a[i] < FLT_MIN) {
            if (a[i] < negative_error) {
                error[0] = 1;
            }
            a[i] = FLT_MIN;
        }
    }
}
__global__ void WAdd(Double* a, Double* b, Double w1, Double w2, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = w1 * a[i] + w2 * b[i];
    }
}
__global__ void WAddTo(Double* dist, Double* b, Double w0, Double w1, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] = w0 * dist[i] + w1 * b[i];
    }
}
__global__ void Add(Double* a, Double* b, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = a[i] + b[i];
    }
}
__global__ void AddTo(Double* dist, Double* b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] += b[i];
    }
}
__global__ void AddTo(Double* dist, Double b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] += b;
    }
}
__global__ void MultiToM(Double* dist, Double* b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] *= b[i];
    }
}
__global__ void MultiTo(Double* dist, Double b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] *= b;
    }
}
__global__ void Minus(Double* a, Double* b, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = a[i] - b[i];
    }
}
__global__ void Multi(Double* a, Double* b, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = a[i] * b[i];
    }
}
__global__ void MultiD2Z(Double* a, CufftComplex* b, CufftComplex* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i].x = a[i] * b[i].x;
        result[i].y = a[i] * b[i].y;
    }
}
__global__ void MultiToD2Z(Double* a, CufftComplex* dist, int N, int V)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ii = i % V;
    if (i < N) {
        dist[i].x *= a[ii];
        dist[i].y *= a[ii];
    }
}
__global__ void MultiToD2D_2d(Double* a, Double* dist, int N, int V)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ii = i % V;
    if (i < N) {
        dist[i] *= a[ii];
    }

}
__global__ void MultiD2D_2d(Double* a, Double* b, Double* result, int N, int V)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ii = i % V;
    if (i < N) {
        result[i] = b[i] * a[ii];
    }
}
__global__ void MultiToK2Z(CufftComplex* dist, Double k, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i].x *= k;
        dist[i].y *= k;
    }
}
__global__ void MultiK2D(Double* a, Double k, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = a[i] * k;
    }
}
__global__ void Init_Array(Double* a, Double value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        a[i] = value;
    }
}
__global__ void Init_ArrayInt(int* a, int value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        a[i] = value;
    }
}
__global__ void Ln(Double* a, Double k, Double* result, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = k * log(a[i]);
    }
}
__global__ void LnTo(Double* dist, Double k, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] = k * log(dist[i]);
    }
}
__global__ void Exp(Double* a, Double k, Double* result, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        result[i] = exp(k * a[i]);
    }
}
__global__ void ExpTo(Double* dist, Double k, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] = exp(k * dist[i]);
    }
}
__global__ void Copy(Double* ori, Double* dist, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] = ori[i];
    }
}
__global__ void CopyInt(int* ori, int* dist, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        dist[i] = ori[i];
    }
}
__global__ void Sum_gpu(Double* a, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        a[i] += a[i + N];
    }
}
__global__ void q_mean_al(Double* q1, Double* q2, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        q2[i] = (4.0 * q1[i] - q2[i]) / 3.0;
    }
}
__global__ void q_mean(Double* q1, Double* q2, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        q2[i] = (q1[i] + q2[i]) / 2.0;
    }
}
__global__ void cal_rho_Ab_multi(Double** q_mid, int chain_num, int s0_num, int V, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int l = i / V;
        int n = i % V;
        q_mid[s0_num + 1][2 * l * V + n] = q_mid[s0_num][2 * l * V + n] * q_mid[s0_num][(2 * l + 1) * V + n];
    }
}
__global__ void cal_rho_Ab_add(Double** q_mid, Double** rhoAb, int start_chain, int L, int s0_num, int V, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ix, iy, iz;
    int jx, jy, jz;
    int j1, j2, j3;
    if (i < N) {
        int l = i / V;
        int n = i % V;
        rhoAb[start_chain + l][n] = -q_mid[s0_num + 1][2 * l * V + n];
        iz = n % L;
        iy = n / L;
        ix = iy / L;
        iy = iy % L;
        for (j1 = -1; j1 < 2; j1++) {
            jx = (ix + j1 + L) % L;
            for (j2 = -1; j2 < 2; j2++) {
                jy = (iy + j2 + L) % L;
                for (j3 = -1; j3 < 2; j3++) {
                    jz = (iz + j3 + L) % L;
                    rhoAb[start_chain + l][n] += q_mid[s0_num + 1][2 * l * V + jx * L * L + jy * L + jz];
                }
            }
        }
        //rhoAb[start_chain + l][n] /= 26.0;
        rhoAb[start_chain + l][n] *= DR3;
    }
}
__global__ void q_multi(Double** q_mid, int chain_num, int V, int s_num, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int j = i / (V * chain_num);
        int k = i % (V * chain_num);
        int l = k / V;
        int n = k % V;
        q_mid[j][2 * l * V + n] *= q_mid[s_num - j - 1][(2 * l + 1) * V + n];
    }
}

__global__ void q_multi_H(Double** q_mid, int cross_num, int V, int s0, int s_N, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int j = i / (s0 * V);
        int k = i % (s0 * V);
        int m = k / V;
        int n = k % V;
        q_mid[s_N - m - 1][j * 2 * V + n + cross_num * V] *= q_mid[m][j * 2 * V + n + V];
        q_mid[s_N - m - 1][j * 2 * V + n + V + cross_num * V] *= q_mid[m][j * 2 * V + n];
        q_mid[s_N - m - 1][j * 2 * V + n + cross_num * V * 2] *= q_mid[m][j * 2 * V + n + V];
        q_mid[s_N - m - 1][j * 2 * V + n + V + cross_num * V * 2] *= q_mid[m][j * 2 * V + n];
    }
}
__global__ void q_sum_init(Double** q_mid, int s0_num, int chain_num, int V, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j, k, l;
    if (i < N) {
        if (i < V * chain_num * s0_num) {
            j = i / (V * chain_num);
            i = i % (V * chain_num);
            k = i / V;
            l = i % V;
            q_mid[j][2 * k * V + l] += q_mid[s0_num * 2 - j - 1][2 * k * V + l];

        }
        else if (i < V * chain_num * s0_num * 2) {
            i -= V * chain_num * s0_num;
            j = i / (V * chain_num);
            i = i % (V * chain_num);
            k = i / V;
            l = i % V;
            q_mid[s0_num + j][2 * k * V + l + chain_num * 2 * V] += q_mid[s0_num + j][(2 * k + 1) * V + l + chain_num * 2 * V];
        }
        else {
            i -= V * chain_num * s0_num * 2;
            j = i / (V * chain_num);
            i = i % (V * chain_num);
            k = i / V;
            l = i % V;
            q_mid[s0_num + j][2 * k * V + l + chain_num * 4 * V] += q_mid[s0_num + j][(2 * k + 1) * V + l + chain_num * 4 * V];
        }
    }
}
__global__ void q_sum_length(Double** q_mid, int s0_num, int length, int chain_num, bool IsOdd, int V, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j, k, l;
    if (i < N) {
        if (i < V * chain_num * length) {
            j = i / (V * chain_num);
            if (j == 0 && IsOdd) {
                return;
            }
            else {
                i = i % (V * chain_num);
                k = i / V;
                l = i % V;
                q_mid[j][2 * k * V + l] += q_mid[j + length][2 * k * V + l];
            }
        }
        else if (i < V * chain_num * length * 2) {
            i -= V * chain_num * length;
            j = i / (V * chain_num);
            if (j == 0 && IsOdd) {
                return;
            }
            else {
                i = i % (V * chain_num);
                k = i / V;
                l = i % V;
                q_mid[s0_num + j][2 * k * V + l + chain_num * 2 * V] += q_mid[s0_num + j + length][2 * k * V + l + chain_num * 2 * V];
            }
        }
        else {
            i -= V * chain_num * length * 2;
            j = i / (V * chain_num);
            if (j == 0 && IsOdd) {
                return;
            }
            else {
                i = i % (V * chain_num);
                k = i / V;
                l = i % V;
                q_mid[s0_num + j][2 * k * V + l + chain_num * 4 * V] += q_mid[s0_num + j + length][2 * k * V + l + chain_num * 4 * V];
            }
        }
    }
}
//__global__ void q_sum(Double* q_temp_sum, int V, int n, int N)
//{
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        q_temp_sum[i] += q_temp_sum[i + V * n];
//    }
//}
__global__ void set_value(Double* a, Double value, int start, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        a[i + start] = value;
    }
}
//__global__ void Cal_fe(Double** rho, Double* temp,Double* eps, Double*WCA,int N)//0AA,1AB,2BB,3AAb,4BAb,5AbAb
//{
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        temp[i] = 0.5 * eps[eps_AA] * rho[rA_id][i] * rho[rA_id][i] + 0.5 * eps[eps_BB] * rho[rB_id][i] * rho[rB_id][i] + 0.5 * eps[eps_AbAb] * (rho[rAbb_id][i] + rho[rAbf_id][i]) * (rho[rAbb_id][i] + rho[rAbf_id][i]);
//        temp[i] = temp[i] + eps[eps_AB] * rho[rA_id][i] * rho[rB_id][i] + eps[eps_AAb] * rho[rA_id][i] * (rho[rAbb_id][i] + rho[rAbf_id][i]) + eps[eps_BAb] * rho[rB_id][i] * (rho[rAbb_id][i] + rho[rAbf_id][i]);
//        temp[i] = temp[i] + (rho[rA_id][i] + rho[rB_id][i] + rho[rAbb_id][i] + rho[rAbf_id][i]) * WCA[i];
//    }
//}
//__global__ void Cal_fsAb(Double** rho, Double** omega, Double* temp, int N) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        temp[i] = -rho[rAbb_id][i] * omega[rAbb_id][i] - rho[rAbf_id][i] * omega[rAbf_id][i];
//    }
//}
//__global__ void Cal_fsNet(Double** rho, Double** omega, Double* temp, int N) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        temp[i] = -rho[rA_id][i] * omega[rA_id][i] - rho[rB_id][i] * omega[rB_id][i];
//    }
//}
//__global__ void Cal_omega_new(Double** rho,  Double* eps,Double* WCA,Double*u,Double** omega_new, int N) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        omega_new[rA_id][i] = eps[eps_AA] * rho[rA_id][i] + eps[eps_AB] * rho[rB_id][i] + eps[eps_AAb] * (rho[rAbf_id][i] + rho[rAbb_id][i])+WCA[i];
//        omega_new[rB_id][i] = eps[eps_AB] * rho[rA_id][i] + eps[eps_BB] * rho[rB_id][i] + eps[eps_BAb] * (rho[rAbf_id][i] + rho[rAbb_id][i])+WCA[i];
//        omega_new[rAbb_id][i] = eps[eps_AAb] * rho[rA_id][i] + eps[eps_BAb] * rho[rB_id][i] + eps[eps_AbAb] * (rho[rAbf_id][i] + rho[rAbb_id][i])+WCA[i]+u[i];
//        omega_new[rAbf_id][i] = eps[eps_AAb] * rho[rA_id][i] + eps[eps_BAb] * rho[rB_id][i] + eps[eps_AbAb] * (rho[rAbf_id][i] + rho[rAbb_id][i])+WCA[i];
//    }
//}
//__global__ void omega_iter(Double** omega, Double** omega_new, Double step,int N) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    if (i < N) {
//        for (int j = 0; j < 4; j++) {
//            omega[j][i] = step * omega_new[j][i] + (1 - step) * omega[j][i];
//        }
//    }
//}
__global__ void link_multi(Double* bij, Double* bij_temp, Double* exp_dG, Double* bi, int N_Ab, int chain_num, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int j = i / N_Ab;
        int k = i % N_Ab;
        bij[i] = exp_dG[i] * bi[j];
        bij_temp[i] = exp_dG[i] * bi[chain_num + k];
    }
}
__global__ void link_add(Double* bij, Double* bij_temp, int N_Ab, int chain_num, bool IsOdd1, bool IsOdd2, int l1, int l2, int N1, int N2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N1) {
        int j = i / N_Ab;
        int k = i % N_Ab;
        if (j == 0 && IsOdd1) {

        }
        else {
            bij[i] += bij[k + (j + l1) * N_Ab];
        }
    }
    else if (i < N2 + N1) {
        i -= N1;
        int j = i / chain_num;
        int k = i % chain_num;
        if (j == 0 && IsOdd2) {

        }
        else {
            bij_temp[k * N_Ab + j] += bij_temp[k * N_Ab + j + l2];
        }
    }
}
__global__ void link_bi(Double* bij, Double* bij_temp, Double* bi, int N_Ab, int chain_num, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        if (i < chain_num) {
            bi[i] = 1.0 / (1.0 + bij_temp[i * N_Ab]);
        }
        else {
            bi[i] = 1.0 / (1.0 + bij[i - chain_num]);
        }
    }
}
__global__ void link_cal_bij(Double* bi, Double* bij, Double* exp_dG, int N_Ab, int chain_num, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int j = i / N_Ab;
        int k = i % N_Ab;
        bij[i] = exp_dG[i] * bi[j] * bi[chain_num + k];
    }
}
__global__ void Cal_bond_delta(Double* fa, Double* fb, int L, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ix, iy, iz;
    int jx, jy, jz;
    int j1, j2, j3;
    if (i < N) {
        fb[i] = -fa[i];
        iz = i % L;
        iy = i / L;
        ix = iy / L;
        iy = iy % L;
        for (j1 = -1; j1 < 2; j1++) {
            jx = (ix + j1 + L) % L;
            for (j2 = -1; j2 < 2; j2++) {
                jy = (iy + j2 + L) % L;
                for (j3 = -1; j3 < 2; j3++) {
                    jz = (iz + j3 + L) % L;
                    fb[i] += fa[jx * L * L + jy * L + jz];
                }
            }
        }
        //fb[i] /= 26.0;
        fb[i] *= DR3;
    }
}
__global__ void Aver_Around3D(Double* fb, Double* fa, int L, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int ix, iy, iz;
    int jx, jy, jz;
    int j1, j2, j3;
    if (i < N) {
        fb[i] = -fa[i];
        iz = i % L;
        iy = i / L;
        ix = iy / L;
        iy = iy % L;
        for (j1 = -1; j1 < 2; j1++) {
            jx = (ix + j1 + L) % L;
            for (j2 = -1; j2 < 2; j2++) {
                jy = (iy + j2 + L) % L;
                for (j3 = -1; j3 < 2; j3++) {
                    jz = (iz + j3 + L) % L;
                    fb[i] += fa[jx * L * L + jy * L + jz];
                }
            }
        }
        //fb[i] /= 26.0;
        fb[i] *= DR3;
    }
}
class ArrayInt {
public:
    ArrayInt(int d, int* dX, bool G);
    ArrayInt(int d, int N, bool G);
    int dim;
    int* shape;
    bool IsGPU;
    int* p;
    int Asize = 1;
    ~ArrayInt();
    int N_blocks;
    void Init(int a);
    void Init(char* filename);
    void Init(ArrayInt& A);
    void aCopy(ArrayInt& A);
    void aSave(char* filename);
    int axis_m2o(int* m);
    void axis_o2m(int o, int* m);
};
ArrayInt::ArrayInt(int d, int N, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = N;
        Asize *= N;
    }
    if (IsGPU) {
        cudaMalloc((void**)&p, sizeof(int) * Asize);
        cudaDeviceSynchronize();
    }
    else {
        p = (int*)malloc(sizeof(int) * Asize);
    }
    N_blocks = (Asize + threads - 1) / threads;
}
ArrayInt::ArrayInt(int d, int* dX, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = dX[i];
        Asize *= dX[i];
    }
    if (IsGPU) {
        cudaMalloc((void**)&p, sizeof(int) * Asize);
        cudaDeviceSynchronize();
    }
    else {
        p = (int*)malloc(sizeof(int) * Asize);
    }
    N_blocks = (Asize + threads - 1) / threads;
}
ArrayInt::~ArrayInt() {
    free(shape);
    if (IsGPU) {
        cudaFree(p);
    }
    else {
        free(p);
    }
}
void ArrayInt::Init(int a) {
    if (IsGPU) {
        Init_ArrayInt << <N_blocks, threads >> > (p, a, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] = a;
        }
    }
}
void ArrayInt::Init(ArrayInt& A) {
    aCopy(A);
}
void ArrayInt::Init(char* filename) {
    ifstream ifs(filename, std::ios::binary | std::ios::in);
    ifs.read((char*)p, sizeof(int) * Asize);
    ifs.close();
}
void ArrayInt::aCopy(ArrayInt& A) {
    int N = min(Asize, A.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU && A.IsGPU) {
        CopyInt << <blocks, threads >> > (A.p, p, N);
        cudaDeviceSynchronize();
    }
    else if (IsGPU) {
        cudaMemcpy(p, A.p, sizeof(int) * N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else if (A.IsGPU) {
        cudaMemcpy(p, A.p, sizeof(int) * N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = A.p[i];
        }
    }
}
void ArrayInt::aSave(char* filename) {
    SaveData(p, filename, Asize);
}
int ArrayInt::axis_m2o(int* m) {
    int axis = 0;
    int temp = 1;
    for (int i = dim - 1; i >= 0; i--) {
        axis = axis + temp * shape[i];
        temp *= shape[i];
    }
    *m = temp;
    return 0;
}
void ArrayInt::axis_o2m(int o, int* m) {
    int temp = 1;
    for (int i = dim - 1; i > 0; i--) {
        temp *= shape[i];
    }
    for (int i = 0; i < dim; i++) {
        m[i] = o / temp;
        o %= temp;
        temp /= shape[i];
    }
    return;
}
class Array {
private:
    Double aSum_reduced(int length);
    int hold = 0;
public:
    Array(int d, int* dX, bool G);
    Array(Double* _p, int d, int* dX, bool G);
    Array(int d, int N, bool G);
    Array(Double* _p, int d, int N, bool G);
    int dim;
    int* shape;
    bool IsGPU;
    Double* p;
    int Asize = 1;
    bool IsVirtual = false;
    ~Array();
    int N_blocks;
    int p_t = 0;
    void Init(Double a);
    void SetValue(Double a, int N = -1, int s0 = 0);
    void Init(char* filename);
    void Init(Array& A);
    void aCopy(Array& A);
    void aCopy(Array& A, int N, int s0 = 0, int s1 = 0);
    void aSave(char* filename);
    void aAdd(Array& A);
    void aAdd(Double a);
    void aAdd(Array& A1, Array& A2);
    void aAdd(Array& A, int N, int s0 = 0, int s1 = 0);
    void aAdd(Double a, int N, int s0 = 0);
    void aAdd(Array& A1, Array& A2, int N, int s0 = 0, int s1 = 0, int s2 = 0);
    void aMulti(Array& A);
    void aMulti(Double a);
    void aMulti(Array& A1, Array& A2);
    void aMulti(Array& A, int N, int s0 = 0, int s1 = 0);
    void aMulti(Double a, int N, int s0 = 0);
    void aMulti(Array& A1, Array& A2, int N, int s0 = 0, int s1 = 0, int s2 = 0);
    void aMulti2D(Array& A1, Array& A2);
    void aMulti2D(Array& A1);
    void aAverage_Around3D(Array& A1);
    Double aSum(int N = -1);
    Double aSum(Array& temp, int N = -1);
    Double Get(int i);
    Double Get(int* m);
    void Sum2D(int axis = 0);
    void Sum2DReduced(int axis, int length);
    void aWAdd(Array& A1, Array& A2, Double w1, Double w2);
    void aWAdd(Array& A1, Double w0, Double w1);
    void aExp(Double k);
    void aExp(Array& A1, Double k);
    void aLn(Double k);
    void aLn(Array& A1, Double k);
    int axis_m2o(int* m);
    void axis_o2m(int o, int* m);
    void Array_hold();
    void Positive_Ensure(int N = -1);
    void Append(Double a);
    void Clear();
    void Change(int i, Double a);
    void Save_temp(char* filename);
    //int error = 0;
    int* error_gpu;
};
Double Array::Get(int i) {
    Double a;
    if (IsGPU) {
        cudaMemcpy(&a, &p[i], sizeof(Double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    else {
        a = p[i];
    }
    return a;
}
Double Array::Get(int* m) {
    int i;
    i = axis_m2o(m);
    return Get(i);
}
void Array::Append(Double a) {
    if (IsGPU) {
        cout << "Array in GPU cannot be appended!" << endl;
        exit(-1);
    }
    else {
        if (p_t >= Asize) {
            cout << "Array append out of size!" << endl;
            exit(-1);
        }
        p[p_t] = a;
        p_t += 1;
    }
}
void Array::Save_temp(char* filename) {
    if (IsGPU) {
        cout << "Array in GPU can not be saved!" << endl;
        exit(-1);
    }
    else {
        if (p_t == 0) {
            cout << "Save empty Array!" << endl;
            exit(-1);
        }
        SaveData(p, filename, p_t);
    }
}
void Array::Clear() {
    p_t = 0;
}
void Array::Change(int i, Double a) {
    if (IsGPU) {
        cout << "Array in GPU can not be changed!" << endl;
        exit(-1);
    }
    if (i >= p_t) {
        cout << "Array change error!" << endl;
        exit(-1);
    }
    p[i] = a;
}
void Array::Sum2DReduced(int axis, int length) {
    if (length == 1) {
        return;
    }
    bool IsOdd = false;
    if (length % 2 == 1) {
        IsOdd = true;
        length++;
    }
    length /= 2;
    int N = length * shape[1 - axis];
    int blocks = (N + threads - 1) / threads;
    Sum2D_gpu << <blocks, threads >> > (p, axis, shape[0], shape[1], IsOdd, length, N);
    cudaDeviceSynchronize();
    Sum2DReduced(axis, length);
}
void Array::Sum2D(int axis) {
    if (IsGPU) {
        Sum2DReduced(axis, shape[axis]);
    }
    else {
        if (axis == 0) {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 1; j < shape[0]; j++) {
                    p[i] += p[i + j * shape[0]];
                }
            }
        }
        else {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 1; j < shape[1]; j++) {
                    p[i * shape[1]] += p[j + i * shape[1]];
                }
            }
        }
    }
}
void Array::Positive_Ensure(int N) {
    if (N < 0) {
        N = Asize;
    }
    if (IsGPU) {
        int blocks = (N + threads - 1) / threads;
        Positive_ensure << <blocks, threads >> > (p, error_gpu, N);
        cudaDeviceSynchronize();
        int a;
        cudaMemcpy(&a, error_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (a == 1) {
            printf("Negative error.");
            exit(-1);
        }
    }
    else {
        for (int i = 0; i < N; i++) {
            if (p[i] < FLT_MIN) {
                if (p[i] < negative_error) {
                    printf("Negative error.");
                    exit(-1);
                }
                p[i] = FLT_MIN;

            }
        }
    }
}
void Array::SetValue(Double a, int N, int s0) {
    if (N < 0) {
        N = Asize;
    }
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Init_Array << <blocks, threads >> > (&p[s0], a, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] = a;
        }
    }
}
void Array::Array_hold() {
    hold++;
}
Array::Array(int d, int* dX, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = dX[i];
        Asize *= dX[i];
    }
    if (IsGPU) {
        cudaMalloc((void**)&p, sizeof(Double) * Asize);
        cudaMalloc((void**)&error_gpu, sizeof(int));
        cudaDeviceSynchronize();
        int a = 0;
        cudaMemcpy(error_gpu, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else {
        p = (Double*)malloc(sizeof(Double) * Asize);
    }
    N_blocks = (Asize + threads - 1) / threads;

}
Array::Array(Double* _p, int d, int* dX, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = dX[i];
        Asize *= dX[i];
    }
    if (IsGPU) {
        cudaMalloc((void**)&error_gpu, sizeof(int));
        cudaDeviceSynchronize();
        int a = 0;
        cudaMemcpy(error_gpu, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    p = _p;
    IsVirtual = true;
    N_blocks = (Asize + threads - 1) / threads;
}
Array::Array(int d, int N, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = N;
        Asize *= N;
    }
    if (IsGPU) {
        cudaMalloc((void**)&p, sizeof(Double) * Asize);
        cudaMalloc((void**)&error_gpu, sizeof(int));
        cudaDeviceSynchronize();
        int a = 0;
        cudaMemcpy(error_gpu, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else {
        p = (Double*)malloc(sizeof(Double) * Asize);
    }
    N_blocks = (Asize + threads - 1) / threads;
}
Array::Array(Double* _p, int d, int N, bool G)
{
    dim = d;
    IsGPU = G;
    shape = (int*)malloc(sizeof(int) * dim);
    IsVirtual = true;
    for (int i = 0; i < dim; i++) {
        shape[i] = N;
        Asize *= N;
    }
    if (IsGPU) {
        cudaMalloc((void**)&error_gpu, sizeof(int));
        cudaDeviceSynchronize();
        int a = 0;
        cudaMemcpy(error_gpu, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    p = _p;
    N_blocks = (Asize + threads - 1) / threads;
}
Array::~Array() {
    free(shape);
    if (!IsVirtual) {
        if (IsGPU) {
            cudaFree(error_gpu);
            cudaFree(p);
        }
        else {
            free(p);
        }
    }
    else {
        if (IsGPU) {
            cudaFree(error_gpu);
        }
    }
}
void Array::Init(Double a) {
    if (IsGPU) {
        Init_Array << <N_blocks, threads >> > (p, a, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] = a;
        }
    }
}
void q_sum_reduction(Double** q_mid, int s0_num, int length, int chain_num) {
    int blocks_n;
    int N_temp;
    if (length % 2 == 0) {
        length /= 2;
        N_temp = length * chain_num * VOLUME * 3;
        blocks_n = (N_temp + threads - 1) / threads;
        q_sum_length << <blocks_n, threads >> > (q_mid, s0_num, length, chain_num, false, VOLUME, N_temp);
        cudaDeviceSynchronize();
    }
    else {
        length = length / 2 + 1;
        N_temp = length * chain_num * VOLUME * 3;
        blocks_n = (N_temp + threads - 1) / threads;
        q_sum_length << <blocks_n, threads >> > (q_mid, s0_num, length, chain_num, true, VOLUME, N_temp);
        cudaDeviceSynchronize();
    }
    if (length == 1) {
        return;
    }
    else {
        q_sum_reduction(q_mid, s0_num, length, chain_num);
    }
}
void q_sum(Double** q_mid, int s0_num, int chain_num)
{
    int N_temp = chain_num * s0_num * VOLUME * 3;
    int blocks_n = (N_temp + threads - 1) / threads;

    q_sum_init << <blocks_n, threads >> > (q_mid, s0_num, chain_num, VOLUME, N_temp);
    cudaDeviceSynchronize();
    q_sum_reduction(q_mid, s0_num, s0_num, chain_num);
}
void link_add_reduced(Double* bij, Double* bij_temp, int N_Ab, int chain_num, int l1, int l2) {
    bool isodd1, isodd2;
    if ((l1 % 2 == 0)) {
        isodd1 = false;
        l1 /= 2;
    }
    else {
        isodd1 = true;
        l1 = l1 / 2 + 1;
    }
    if ((l2 % 2 == 0)) {
        isodd2 = false;
        l2 /= 2;
    }
    else {
        isodd2 = true;
        l2 = l2 / 2 + 1;
    }
    int N1 = l1 * N_Ab;
    int N2 = l2 * chain_num;
    int blocks_n = (N1 + N2 + threads - 1) / threads;
    link_add << <blocks_n, threads >> > (bij, bij_temp, N_Ab, chain_num, isodd1, isodd2, l1, l2, N1, N2);
    cudaDeviceSynchronize();
    if (l1 == 1) {
        l1 = 0;
    }
    if (l2 == 1) {
        l2 = 0;
    }
    if (l1 == 0 && l2 == 0) {
        return;
    }
    else {
        link_add_reduced(bij, bij_temp, N_Ab, chain_num, l1, l2);
    }
}
void Cal_link_p(Array* exp_dG, Array* bij, Array* bij_temp, Array* bi, int chain_num, int N_Ab, int iteration) {
    int blocks_n1, blocks_n2;
    int N_temp1 = N_Ab + chain_num;
    int N_temp2 = N_Ab * chain_num;
    blocks_n2 = (N_temp2 + threads - 1) / threads;
    blocks_n1 = (N_temp1 + threads - 1) / threads;
    set_value << <blocks_n1, threads >> > (bi->p, 0.9, 0, N_temp1);
    cudaDeviceSynchronize();
    cout << "bij_iter: " << iteration << endl;
    //int shape[] = { 81,100 };
    //Array debug_cpu(2, shape, false);
    for (int i = 0; i < iteration; i++) {
        link_multi << <blocks_n2, threads >> > (bij->p, bij_temp->p, exp_dG->p, bi->p, N_Ab, chain_num, N_temp2);
        cudaDeviceSynchronize();
        /* debug_cpu.aCopy(*bij);
         SaveData(debug_cpu.p, "b1.dat", 81 * 100);
         debug_cpu.aCopy(*bij_temp);
         SaveData(debug_cpu.p, "b2.dat", 81 * 100);*/
         //link_add_reduced(bij, bij_temp, N_Ab, chain_num, chain_num, N_Ab);
        bij->Sum2D(0);
        bij_temp->Sum2D(1);
        link_bi << <blocks_n1, threads >> > (bij->p, bij_temp->p, bi->p, N_Ab, chain_num, N_temp1);
        //debug_cpu.aCopy(*bi);
        //SaveData(debug_cpu.p, "b3.dat", 81 + 100);
        cudaDeviceSynchronize();

    }
    link_cal_bij << <blocks_n2, threads >> > (bi->p, bij->p, exp_dG->p, N_Ab, chain_num, N_temp2);
    cudaDeviceSynchronize();
}
void Array::Init(Array& A) {
    aCopy(A);
}
void Array::Init(char* filename) {
    ifstream ifs(filename, std::ios::binary | std::ios::in);
    ifs.read((char*)p, sizeof(Double) * Asize);
    ifs.close();
}
void Array::aCopy(Array& A) {
    int N = min(Asize, A.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU && A.IsGPU) {
        Copy << <blocks, threads >> > (A.p, p, N);
        cudaDeviceSynchronize();
    }
    else if (IsGPU) {
        cudaMemcpy(p, A.p, sizeof(Double) * N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else if (A.IsGPU) {
        cudaMemcpy(p, A.p, sizeof(Double) * N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = A.p[i];
        }
    }
}
void Array::aCopy(Array& A, int N, int s0, int s1) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU && A.IsGPU) {
        Copy << <n_blocks, threads >> > (&A.p[s1], &p[s0], N);
        cudaDeviceSynchronize();
    }
    else if (IsGPU) {
        cudaMemcpy(&p[s0], &A.p[s1], sizeof(Double) * N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else if (A.IsGPU) {
        cudaMemcpy(&p[s0], &A.p[s1], sizeof(Double) * N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] = A.p[i + s1];
        }
    }
}
void Array::aSave(char* filename) {
    SaveData(p, filename, Asize);
}
void Array::aAdd(Double a) {

    if (IsGPU) {
        AddTo << <N_blocks, threads >> > (p, a, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] += a;
        }
    }
}
void Array::aAdd(Array& A) {
    int N = min(Asize, A.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        AddTo << <blocks, threads >> > (p, A.p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] += A.p[i];
        }
    }
}
void Array::aAdd(Array& A1, Array& A2) {
    int N = min(Asize, min(A2.Asize, A1.Asize));
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Add << <blocks, threads >> > (A1.p, A2.p, p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = A1.p[i] + A2.p[i];
        }
    }
}
void Array::aAdd(Double a, int N, int s0) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        AddTo << <n_blocks, threads >> > (&p[s0], a, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] += a;
        }
    }
}
void Array::aAdd(Array& A, int N, int s0, int s1) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        AddTo << <n_blocks, threads >> > (&p[s0], &A.p[s1], N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] += A.p[i + s1];
        }
    }
}
void Array::aAdd(Array& A1, Array& A2, int N, int s0, int s1, int s2) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Add << <n_blocks, threads >> > (&A1.p[s1], &A2.p[s2], &p[s0], N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] = A1.p[i + s1] + A2.p[i + s2];
        }
    }
}
void Array::aMulti(Double a) {

    if (IsGPU) {
        MultiTo << <N_blocks, threads >> > (p, a, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] *= a;
        }
    }
}
void Array::aMulti(Array& A) {
    int N = min(Asize, A.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {

        MultiToM << <blocks, threads >> > (p, A.p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] *= A.p[i];
        }
    }
}
void Array::aMulti(Array& A1, Array& A2) {
    int N = min(Asize, min(A2.Asize, A1.Asize));
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Multi << <blocks, threads >> > (A1.p, A2.p, p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = A1.p[i] * A2.p[i];
        }
    }
}
void Array::aMulti(Double a, int N, int s0) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        MultiTo << <n_blocks, threads >> > (&p[s0], a, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] *= a;
        }
    }
}
void Array::aMulti(Array& A, int N, int s0, int s1) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {

        MultiToM << <n_blocks, threads >> > (&p[s0], &A.p[s1], N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] *= A.p[i + s1];
        }
    }
}
void Array::aMulti(Array& A1, Array& A2, int N, int s0, int s1, int s2) {
    int n_blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Multi << <n_blocks, threads >> > (&A1.p[s1], &A2.p[s2], &p[s0], N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i + s0] = A1.p[i + s1] * A2.p[i + s2];
        }
    }
}
void Array::aMulti2D(Array& A1, Array& A2) {
    int blocks = (Asize + threads - 1) / threads;
    if (IsGPU) {
        MultiD2D_2d << <blocks, threads >> > (A2.p, A1.p, p, Asize, A2.Asize);
        cudaDeviceSynchronize();
    }
    else {
        int ii;
        for (int i = 0; i < Asize; i++) {
            ii = i % A2.Asize;
            p[i] = A1.p[i] * A2.p[ii];
        }
    }
}
void Array::aMulti2D(Array& A1) {
    int blocks = (Asize + threads - 1) / threads;
    if (IsGPU) {
        MultiToD2D_2d << <blocks, threads >> > (A1.p, p, Asize, A1.Asize);
        cudaDeviceSynchronize();
    }
    else {
        int ii;
        for (int i = 0; i < Asize; i++) {
            ii = i % A1.Asize;
            p[i] *= A1.p[ii];
        }
    }
}
void Array::aWAdd(Array& A1, Array& A2, Double w1, Double w2) {
    int N = min(Asize, min(A2.Asize, A1.Asize));
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        WAdd << <blocks, threads >> > (A1.p, A2.p, w1, w2, p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = w1 * A1.p[i] + w2 * A2.p[i];
        }
    }
}
void Array::aWAdd(Array& A1, Double w0, Double w1) {
    int N = min(Asize, A1.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        WAddTo << <blocks, threads >> > (p, A1.p, w0, w1, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = w1 * A1.p[i] + w0 * p[i];
        }
    }
}
Double Array::aSum(int N) {
    if (N < 0) {
        N = Asize;
    }
    if (IsGPU) {
        return aSum_reduced(N);
    }
    else {
        Double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += p[i];
        }
        return sum;
    }

}
Double Array::aSum(Array& temp, int N) {
    if (N < 0) {
        N = Asize;
    }
    temp.aCopy(*this, N, 0, 0);
    return temp.aSum(N);
}
Double Array::aSum_reduced(int length) {
    if (length == 1) {

        Double sum;
        cudaMemcpy(&sum, &p[0], sizeof(Double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        return sum;
    }
    if (length % 2 == 0) {
        length /= 2;
        int blocks = (length + threads - 1) / threads;
        Sum_gpu << <blocks, threads >> > (p, length);
        cudaDeviceSynchronize();
    }
    else {
        length = (length - 1) / 2;
        int blocks = (length + threads - 1) / threads;
        Sum_gpu << <blocks, threads >> > (&p[1], length);
        cudaDeviceSynchronize();
        length++;
    }
    return aSum_reduced(length);
}
void Array::aExp(Double k) {
    if (IsGPU) {
        ExpTo << <N_blocks, threads >> > (p, k, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] = exp(p[i] * k);
        }
    }
}
void Array::aExp(Array& A1, Double k) {
    int N = min(Asize, A1.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Exp << <blocks, threads >> > (A1.p, k, p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = exp(A1.p[i] * k);
        }
    }
}
void Array::aLn(Double k) {
    if (IsGPU) {
        LnTo << <N_blocks, threads >> > (p, k, Asize);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < Asize; i++) {
            p[i] = k * log(p[i]);
        }
    }
}
void Array::aLn(Array& A1, Double k) {
    int N = min(Asize, A1.Asize);
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Ln << <blocks, threads >> > (A1.p, k, p, N);
        cudaDeviceSynchronize();
    }
    else {
        for (int i = 0; i < N; i++) {
            p[i] = k * log(A1.p[i]);
        }
    }
}
void Array::aAverage_Around3D(Array& A1) {
    int N, L;
    if (A1.Asize < Asize) {
        N = A1.Asize;
        L = A1.shape[0];
    }
    else {
        N = this->Asize;
        L = this->shape[0];
    }
    int blocks = (N + threads - 1) / threads;
    if (IsGPU) {
        Aver_Around3D << <blocks, threads >> > (this->p, A1.p, L, N);
        cudaDeviceSynchronize();
    }
    else
    {
        int ix, iy, iz;
        int jx, jy, jz;
        int j1, j2, j3;
        for (int i = 0; i < N; i++) {
            this->p[i] = -A1.p[i];
            iz = i % L;
            iy = i / L;
            ix = iy / L;
            iy = iy % L;
            for (j1 = -1; j1 < 2; j1++) {
                jx = (ix + j1 + L) % L;
                for (j2 = -1; j2 < 2; j2++) {
                    jy = (iy + j2 + L) % L;
                    for (j3 = -1; j3 < 2; j3++) {
                        jz = (iz + j3 + L) % L;
                        this->p[i] += A1.p[jx * L * L + jy * L + jz];
                    }
                }
            }
            //this->p[i] /= 26.0;
            this->p[i] *= DR3;
        }
    }
}
int Array::axis_m2o(int* m) {
    int axis = 0;
    int temp = 1;
    for (int i = dim - 1; i >= 0; i--) {
        axis = axis + temp * m[i];
        temp *= shape[i];
    }
    return axis;
}
void Array::axis_o2m(int o, int* m) {
    int temp = 1;
    for (int i = dim - 1; i > 0; i--) {
        temp *= shape[i];
    }
    for (int i = 0; i < dim; i++) {
        m[i] = o / temp;
        o %= temp;
        temp /= shape[i];
    }
    return;
}
class DE_step {
public:
    cufftHandle planf;
    cufftHandle planb;
    int rank, istride, idist, ostride;
    int odist, batch;
    int V = 1;
    int Vm;
    int N_blocks;
    Double k_inverse;
    int* n;
    int* inembed;
    int* onembed;
    CufftComplex* qf;
    Array* q_temp;
    Array* k_fft;
    Array* k2_fft;
    void Init(int _rank, int _istride, int _idist, int _ostride, int _odist, int _batch, int* _n, int* _onembed, int* _inembed);
    void Set_temp_var(CufftComplex* _qf, Array* _q_temp, Array* _k_fft, Array* _k2_fft);
    void Run_a_step(Array** q_mid, Array* omega1, Array* omega2, int start, int* eta_s);
    ~DE_step();
};
void DE_step::Run_a_step(Array** q_mid, Array* omega1, Array* omega2, int start, int* eta_s) {
    //Cuda_err_print();
    q_mid[start + 1]->aMulti2D(*q_mid[start], omega1[eta_s[start]]);
    // MultiD2D_2d << <N_blocks, threads_per_block >> > (omega1[eta_s[i]], q_mid[i], q_mid[i + 1], vm, V);


     //Cuda_err_print();
    CufftExcForward(planf, q_mid[start + 1]->p, qf);
    cudaDeviceSynchronize();
    MultiToD2Z << <N_blocks, threads >> > (k_fft->p, qf, Vm, V);
    cudaDeviceSynchronize();
    CufftExcBackward(planb, qf, q_mid[start + 1]->p);
    cudaDeviceSynchronize();
    q_mid[start + 1]->aMulti(k_inverse);
    //MultiTo << <N_blocks, threads_per_block >> > (q_mid[i + 1], k_inverse, vm);
    q_mid[start + 1]->aMulti2D(omega1[eta_s[start]]);
    //MultiToD2D_2d << <N_blocks, threads_per_block >> > (omega1[eta_s[i]], q_mid[i + 1], vm, V);
    q_temp->aCopy(*q_mid[start]);

    //Copy << <N_blocks, threads_per_block >> > (q_mid[i], q_temp, vm);
    for (int j = 0; j < 2; j++)
    {

        q_temp->aMulti2D(omega2[eta_s[start]]);

        //MultiToD2D_2d << <N_blocks, threads >> > (omega2[eta_s[i]], q_temp, vm, V);
        CufftExcForward(planf, q_temp->p, qf);
        cudaDeviceSynchronize();
        MultiToD2Z << <N_blocks, threads >> > (k2_fft->p, qf, Vm, V);
        cudaDeviceSynchronize();
        CufftExcBackward(planb, qf, q_temp->p);
        cudaDeviceSynchronize();
        q_temp->aMulti(k_inverse);
        //MultiTo << <N_blocks, threads_per_block >> > (q_temp, k_inverse, vm);
        q_temp->aMulti2D(omega2[eta_s[start]]);

        //MultiToD2D_2d << <N_blocks, threads_per_block >> > (omega2[eta_s[i]], q_temp, vm, V);
    }
    //q_mean_al << <N_blocks, threads_per_block >> > (q_temp, q_mid[i + 1], vm);

    q_mid[start + 1]->aWAdd(*q_temp, -1.0 / 3.0, 4.0 / 3.0);
    q_mid[start]->aWAdd(*q_mid[start + 1], 0.5, 0.5);

    //q_mean << <N_blocks, threads_per_block >> > (q_mid[i + 1], q_mid[i], vm);
}
void DE_step::Init(int _rank, int _istride, int _idist, int _ostride, int _odist, int _batch, int* _n, int* _onembed, int* _inembed) {
    n = (int*)malloc(sizeof(int) * _rank);
    inembed = (int*)malloc(sizeof(int) * _rank);
    onembed = (int*)malloc(sizeof(int) * _rank);
    for (int i = 0; i < _rank; i++) {
        n[i] = _n[i];
        inembed[i] = _inembed[i];
        onembed[i] = _onembed[i];
        V *= _n[i];
    }
    k_inverse = 1.0 / Double(V);
    rank = _rank;
    batch = _batch;
    Vm = V * _batch;
    istride = _istride;
    ostride = _ostride;
    idist = _idist;
    odist = _odist;
    cufftPlanMany(&planf, rank, n, inembed, istride, idist, onembed, ostride, odist, CufftForward, batch);
    cudaDeviceSynchronize();
    cufftPlanMany(&planb, rank, n, inembed, istride, idist, onembed, ostride, odist, CufftBackward, batch);
    cudaDeviceSynchronize();
    N_blocks = (Vm + threads - 1) / threads;
}
void DE_step::Set_temp_var(CufftComplex* _qf, Array* _q_temp, Array* _k_fft, Array* _k2_fft) {
    qf = _qf;
    q_temp = _q_temp;
    k_fft = _k_fft;
    k2_fft = _k2_fft;
}
DE_step::~DE_step() {
    cufftDestroy(planf);
    cufftDestroy(planb);
    free(n);
    free(inembed);
    free(onembed);
}

//class Component {
//public:
//    virtual void Normalize_rho() = 0;
//    virtual ~Component() {}
//    virtual void Output(char* filename, bool Isrho, bool Isomega) = 0;
//    virtual void Init() = 0;
//
//};
class Chain;
class NanoParticle {
public:
    Double d;
    Array* WCA;
    Array* U_bond;
    int M;
};
class Antibody {
public:
    int Num;
    Double n0, n1;
    //int box_N;
    //int box_V;
    //Double box_len;
    char* self_name = "Ab";
    //void Output(char* filename, bool Isrho, bool Isomega);
    //void Normalize_rho();
    //void Init(int n, int bn, Double _dr, Double bl);
    void Cal_Z(NanoParticle& NP);
    Array* rho0;
    Array* rho1;
    Array** rho2;
    Array** rho3;
    //Array* rho2_t;
    //Array* rho3_t;
    Array* temp;
    Array* omega;//equal to Net.omega
    Array* rho;
    //Array* Pi;
    Double** rho2_p_gpu;
    Double Z0, Zb;
    ~Antibody();
};
void Antibody::Cal_Z(NanoParticle& NP) {
    rho0->aExp(*omega, -1.0);
    Z0 = rho0->aSum(*temp, omega->Asize);
    Z0 *= DR3;
    rho0->aMulti(1.0 / Z0);
    rho1->aAdd(*omega, *NP.U_bond);
    rho1->aExp(-1.0);
    Zb = rho1->aSum(*temp, omega->Asize);
    Zb *= DR3;
    rho1->aMulti(1.0 / Zb);
}
//void Antibody::Init(int n, int bn, Double _dr, Double bl) {
//    Num = n;
//    box_N = bn;
//    box_V = bn * bn * bn;
//    box_len = bl;
//    for (int i = 0; i < 4; i++) {
//        //rho[i] = new Array(3, Num, true);
//    }
//}
Antibody::~Antibody() {
    //for (int i = 0; i < 4; i++) {
    //    //delete rho[i];
    //}
}
//class Solvent{
//public:
//    int Num;
//    Array* rho;
//    Array* omega;
//};
class CrossPoint {
public:
    int id;
    Array* p;
    Array* h;
    Array* p_n;
    Array* h_n;
    Array* temp;
    Array* p_all[3];
    Double int_p_h[3];
    Double Z;
    bool Isfree;
    void h2p();
    void Init(int N, int V);
    void Update_h(Array& n_p, Double step);
    Double Cal_H(Array& n_p);
    Double Cal_Z();
    ~CrossPoint();
};
void CrossPoint::Init(int N, int V) {
    p = new Array(3, N, true);
    p_n = p;
    h = new Array(3, N, true);
}
CrossPoint::~CrossPoint() {
    delete p;
    delete h;
}

class Chain {
public:
    int id;
    CrossPoint* c1;
    CrossPoint* c2;
    Array** rho;
    Array* Qi, * Qj, * Z1i, * Z1j, * Z2i, * Z2j;
    Double Qij, Z1ij, Z2ij;
    Array* Zi012[3];
    Array* Zj012[3];
    Array* Pi, ** p_i_m;
    void Init(int N, int V, int max_bond);
    ~Chain();

};
Chain::~Chain() {
    delete c1;
    delete c2;
    for (int i = 0; i < 3; i++) {
        delete rho[i];
    }
    delete rho;
    delete Qi;
    delete Qj;
    delete Z1i;
    delete Z1j;
    delete Z2i;
    delete Z2j;
    delete Pi;
    delete p_i_m[0];
    delete p_i_m[1];
    delete p_i_m[2];
    delete p_i_m;

}
void Chain::Init(int N, int V, int max_bond) {
    c1 = new CrossPoint;
    c2 = new CrossPoint;
    rho = new Array * [3];
    for (int i = 0; i < 3; i++) {
        rho[i] = new Array(3, N, true);
    }
    Qi = new Array(3, N, true);
    Qj = new Array(3, N, true);
    Z1i = new Array(3, N, true);
    Z1j = new Array(3, N, true);
    Z2i = new Array(3, N, true);
    Z2j = new Array(3, N, true);
    Pi = new Array(1, 3, false);
    p_i_m = new Array * [3];
    p_i_m[0] = new Array(1, max_bond + 1, false);
    p_i_m[1] = new Array(1, max_bond + 1, false);
    p_i_m[2] = new Array(1, max_bond + 1, false);
    Zi012[0] = Qi;
    Zi012[1] = Z1i;
    Zi012[2] = Z2i;
    Zj012[0] = Qj;
    Zj012[1] = Z1j;
    Zj012[2] = Z2j;
    c1->Init(N, V);
    c2->Init(N, V);
}
class Network {
public:
    int chain_num, cross_num;
    //CrossPoint* cross_to_cross;
    Chain*** cross_to_chain;
    Chain* chain;
    int s0;
    int s_N;
    int* eta_s;
    Array* rho;
    Array* omega;
    //Array* P_cross;
    Array* omega_n;
};

class SCF_system {
public:
    NanoParticle NP;
    Network Net;
    Antibody Ab;
    Array** q_mid;
    Array* omega_exp1;
    Array* omega_exp2;
    Array* h1;
    Array* h2;
    Array* temp;
    Array* t1, * t2, * t3, * t4, * t5;
    Array* dGij, * bij, * bij_temp;
    Array* dG_cnf1, * dG_cnf2;
    Array* bi, * bi_cpu;
    Array* cpu_temp;
    Double** q_mid_p_gpu;
    DE_step DE1, DE2;
    int batch_size;
    int Volume;
    int box_N;
    int bij_iter;
    int H1_start, H2_start;
    char string_temp[50];
    bool IsUnbind;
    void Init();
    void DE_Init();
    void Temp_var_Init();
    void Net_init();
    void Ab_Init();
    void NP_Init();
    void Save_var(char* filename, Array& A);
    //void Set_Parameter(int _box_n,int _bat_size,Double _ds,Double _step,Double _v0,Double dG_Net,Double dG_NP);
    void Run_an_iter(bool output = false);
    void Cal_Zm(int m);
    void z_cross_multi(int& ni, Double& p1, Double& p2, CrossPoint* c, int i2, Array* _t1, Array* _t2, int j);
    void Output_data(char* file_name_ex);
    int iteration;
    int bond_max_num;
    Double ds, step;
    Double v0;
    Double dG_Net, dG_NP;
    Array* P_unbind_m;
    Double P_unbind;
    Array* ln_Hm;
    Double ln_Z, F;
    Double* mem_p;
    Double* out_var;
    int it = 0;
    ~SCF_system();
};
SCF_system::~SCF_system() {
    delete cpu_temp;
    delete DE1.k_fft;
    delete DE1.k2_fft;
    cudaFree(DE1.qf);
    delete DE1.q_temp;
    delete DE2.q_temp;
    cudaFree(q_mid_p_gpu);
    for (int i = 0; i < Net.s_N + 1; i++) {
        delete q_mid[i];
    }
    delete q_mid;
    delete temp;
    delete t1;
    delete t2;
    delete t3;
    delete t4;
    delete t5;
    delete Net.omega;
    delete Net.rho;
    delete Net.eta_s;
    for (int j = 0; j < Net.cross_num; j++) {
        delete  Net.cross_to_chain[j];
    }
    delete Net.cross_to_chain;
    delete[] Net.chain;
    delete NP.WCA;
    delete NP.U_bond;
    //delete Ab.omega;
    delete Ab.rho;
    delete Ab.rho0;
    delete Ab.rho1;
    for (int j = 0; j < Net.chain_num; j++) {
        delete Ab.rho2[j];
        delete Ab.rho3[j];
    }
    delete[] Ab.rho2;
    cudaFree(Ab.rho2_p_gpu);
    delete[] Ab.rho3;
    delete omega_exp1;
    delete omega_exp2;
    delete h1;
    delete h2;
    delete dGij;
    delete bij;
    delete bij_temp;
    delete dG_cnf1;
    delete dG_cnf2;
    delete bi;
    delete bi_cpu;
    delete P_unbind_m;
    delete ln_Hm;
    delete out_var;
}


void SCF_system::DE_Init() {
    if (IsDouble) {
        ifstream ifs("k_fft_D.dat", std::ios::binary | std::ios::in);
        ifs.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs.close();
    }
    else {
        ifstream ifs("k_fft.dat", std::ios::binary | std::ios::in);
        ifs.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs.close();
    }

    DE1.k_fft = new Array(3, box_N, true);
    DE1.k_fft->aCopy(*cpu_temp);
    DE2.k_fft = DE1.k_fft;
    if (IsDouble) {
        ifstream ifs2("k2_fft_D.dat", std::ios::binary | std::ios::in);
        ifs2.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs2.close();
    }
    else {
        ifstream ifs2("k2_fft.dat", std::ios::binary | std::ios::in);
        ifs2.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs2.close();
    }

    DE1.k2_fft = new Array(3, box_N, true);
    DE1.k2_fft->aCopy(*cpu_temp);
    DE2.k2_fft = DE1.k2_fft;
    int* t = new int[3];
    for (int i = 0; i < 3; i++) {
        t[i] = box_N;
    }

    DE1.Init(3, 1, Volume, 1, Volume, batch_size * 2, t, t, t);

    DE2.Init(3, 1, Volume, 1, Volume, batch_size * 2 * 3, t, t, t);

    delete t;
    cudaMalloc((void**)&DE2.qf, Volume * sizeof(CufftComplex) * batch_size * 2 * 3);
    cudaDeviceSynchronize();
    DE1.qf = DE2.qf;
    int shape[4];
    shape[0] = batch_size * 2 * 3;
    shape[1] = box_N;
    shape[2] = box_N;
    shape[3] = box_N;
    DE2.q_temp = new Array(4, shape, true);
    mem_p = DE2.q_temp->p;
    shape[0] = batch_size * 2;
    DE1.q_temp = new Array(mem_p, 4, shape, true);
    q_mid = new Array * [Net.s_N + 1];
    shape[0] = batch_size * 2;
    cudaMalloc((void**)&q_mid_p_gpu, (Net.s_N + 1) * sizeof(Double*));
    cudaDeviceSynchronize();
    for (int i = 0; i < Net.s0; i++) {
        q_mid[i] = new Array(4, shape, true);
        cudaMemcpy(&q_mid_p_gpu[i], &q_mid[i]->p, sizeof(Double*), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    shape[0] = batch_size * 2 * 3;
    for (int i = Net.s0; i < Net.s_N + 1; i++) {
        q_mid[i] = new Array(4, shape, true);
        cudaMemcpy(&q_mid_p_gpu[i], &q_mid[i]->p, sizeof(Double*), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

}
void SCF_system::Temp_var_Init() {
    temp = new Array(mem_p, 3, box_N, true);
    t1 = new Array(&mem_p[Volume], 3, box_N, true);
    t2 = new Array(&mem_p[Volume * 2], 3, box_N, true);
    t3 = new Array(&mem_p[Volume * 3], 3, box_N, true);
    t4 = new Array(&mem_p[Volume * 4], 3, box_N, true);
    t5 = new Array(&mem_p[Volume * 5], 3, box_N, true);
}
void SCF_system::Net_init() {
    Net.omega_n = temp;
    if (IsDouble) {
        ifstream ifs3("omega_Net_init_D.dat", std::ios::binary | std::ios::in);
        ifs3.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs3.close();
    }
    else {
        ifstream ifs3("omega_Net_init.dat", std::ios::binary | std::ios::in);
        ifs3.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs3.close();
    }

    Net.omega = new Array(3, box_N, true);
    Net.rho = new Array(3, box_N, true);
    Net.omega->aCopy(*cpu_temp);
    Net.eta_s = new int[Net.s_N];
    ifstream ifs8("eta_s.dat", std::ios::binary | std::ios::in);
    ifs8.read((char*)Net.eta_s, sizeof(int) * Net.s_N);
    ifs8.close();
    Net.chain = new Chain[Net.chain_num];
    int* chains = new int[Net.chain_num * 2];
    int* Is_free = new int[Net.cross_num];
    ifstream ifs14("chains.dat", std::ios::binary | std::ios::in);
    ifs14.read((char*)chains, sizeof(int) * Net.chain_num * 2);
    ifs14.close();
    ifstream ifs15("cross_free.dat", std::ios::binary | std::ios::in);
    ifs15.read((char*)Is_free, sizeof(int) * Net.cross_num);
    ifs15.close();
    Chain* ct;
    char filename[50];
    Net.cross_to_chain = new Chain * *[Net.cross_num];
    int* len = new int[Net.cross_num];

    for (int j = 0; j < Net.cross_num; j++) {
        Net.cross_to_chain[j] = new Chain * [6];
        len[j] = 0;
    }

    for (int j = 0; j < Net.chain_num; j++) {
        ct = &Net.chain[j];
        ct->Init(box_N, Volume, bond_max_num);
        ct->id = j;
        ct->c1->id = chains[j * 2];
        ct->c2->id = chains[j * 2 + 1];
        if (Is_free[ct->c1->id]) {
            ct->c1->Isfree = true;
        }
        else {
            ct->c1->Isfree = false;
        }
        if (Is_free[ct->c2->id]) {
            ct->c2->Isfree = true;
        }
        else {
            ct->c2->Isfree = false;
        }

        Net.cross_to_chain[ct->c1->id][len[ct->c1->id]] = ct;
        len[ct->c1->id]++;
        Net.cross_to_chain[ct->c2->id][len[ct->c2->id]] = ct;
        len[ct->c2->id]++;
        ct->c1->h_n = t3;
        ct->c1->temp = temp;
        ct->c1->p_all[0] = t1;
        ct->c1->p_all[1] = t1;
        ct->c1->p_all[2] = t1;
        ct->c2->temp = temp;
        ct->c2->h_n = t4;
        ct->c2->p_all[0] = t2;
        ct->c2->p_all[1] = t2;
        ct->c2->p_all[2] = t2;
        if (IsDouble) {
            sprintf(filename, "h1_%d_D.dat", j);
            ifstream ifs00(filename, std::ios::binary | std::ios::in);
            ifs00.read((char*)cpu_temp->p, sizeof(Double) * Volume);
            ifs00.close();
            ct->c1->h->aCopy(*cpu_temp);
            sprintf(filename, "h2_%d_D.dat", j);
            ifstream ifs1(filename, std::ios::binary | std::ios::in);
            ifs1.read((char*)cpu_temp->p, sizeof(Double) * Volume);
            ifs1.close();
            ct->c2->h->aCopy(*cpu_temp);
        }
        else {
            sprintf(filename, "h1_%d.dat", j);
            ifstream ifs00(filename, std::ios::binary | std::ios::in);
            ifs00.read((char*)cpu_temp->p, sizeof(Double) * Volume);
            ifs00.close();
            ct->c1->h->aCopy(*cpu_temp);
            sprintf(filename, "h2_%d.dat", j);
            ifstream ifs1(filename, std::ios::binary | std::ios::in);
            ifs1.read((char*)cpu_temp->p, sizeof(Double) * Volume);
            ifs1.close();
            ct->c2->h->aCopy(*cpu_temp);
        }


    }

    Chain* chain_temp;
    for (int j = 0; j < Net.chain_num; j++) {
        chain_temp = &Net.chain[j];

        chain_temp->c1->Cal_Z();

        chain_temp->c1->p->aExp(*chain_temp->c1->h, -1.0);
        chain_temp->c1->p->aMulti(1.0 / chain_temp->c1->Z);
        chain_temp->c2->Cal_Z();
        chain_temp->c2->p->aExp(*chain_temp->c2->h, -1.0);
        chain_temp->c2->p->aMulti(1.0 / chain_temp->c2->Z);
    }

    delete chains;
    delete len;
    delete Is_free;
}
void SCF_system::NP_Init() {

    if (IsDouble) {
        ifstream ifs12("WCA_D.dat", std::ios::binary | std::ios::in);
        ifs12.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs12.close();
    }
    else {
        ifstream ifs12("WCA.dat", std::ios::binary | std::ios::in);
        ifs12.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs12.close();
    }

    NP.WCA = new Array(3, box_N, true);
    NP.WCA->aCopy(*cpu_temp);
    if (IsDouble) {
        ifstream ifs13("u_D.dat", std::ios::binary | std::ios::in);
        ifs13.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs13.close();
    }
    else {
        ifstream ifs13("u.dat", std::ios::binary | std::ios::in);
        ifs13.read((char*)cpu_temp->p, sizeof(Double) * Volume);
        ifs13.close();
    }

    NP.U_bond = new Array(3, box_N, true);
    NP.U_bond->aCopy(*cpu_temp);
}
void SCF_system::Ab_Init() {
    //ifstream ifs5("omega_Ab_init.dat", std::ios::binary | std::ios::in);
    //ifs5.read((char*)cpu_temp->p, sizeof(Double) * Volume);
    //ifs5.close();
    //Ab.omega = new Array(3, box_N, true);
    //Ab.omega->aCopy(*cpu_temp);

    Ab.omega = Net.omega;
    Ab.temp = temp;
    Ab.rho = new Array(3, box_N, true);
    Ab.rho0 = new Array(3, box_N, true);
    Ab.rho1 = new Array(3, box_N, true);
    cudaMalloc((void**)&Ab.rho2_p_gpu, Net.chain_num * sizeof(Double*));
    cudaDeviceSynchronize();
    Ab.rho2 = new Array * [Net.chain_num];
    Ab.rho3 = new Array * [Net.chain_num];
    for (int j = 0; j < Net.chain_num; j++) {
        Ab.rho2[j] = new Array(3, box_N, true);
        Ab.rho3[j] = new Array(3, box_N, true);
        cudaMemcpy(&Ab.rho2_p_gpu[j], &Ab.rho2[j]->p, sizeof(Double*), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    Ab.Cal_Z(NP);
}
void SCF_system::Init() {
    cpu_temp = new Array(3, box_N, false);
    bond_max_num = min(Ab.Num, NP.M);

    DE_Init();

    Temp_var_Init();

    Net_init();

    NP_Init();
    Ab_Init();

    omega_exp1 = new Array(3, box_N, true);
    omega_exp2 = new Array(3, box_N, true);
    h1 = new Array(3, box_N, true);
    h2 = new Array(3, box_N, true);
    H1_start = Volume * batch_size * 2;
    H2_start = Volume * batch_size * 2 * 2;
    if (Net.chain_num * Ab.Num > 6 * batch_size * Volume) {
        printf("memerory error: dGij ");
        exit(0);
    }
    else {
        int shape[2];
        shape[0] = Net.chain_num;
        shape[1] = Ab.Num;
        dGij = new Array(q_mid[Net.s0 + 1]->p, 2, shape, true);
        bij = new Array(q_mid[Net.s0 + 2]->p, 2, shape, true);
        bij_temp = new Array(q_mid[Net.s0 + 3]->p, 2, shape, true);
        dG_cnf1 = new Array(1, Net.chain_num, false);
        dG_cnf2 = new Array(1, Net.chain_num, false);
        bi_cpu = new Array(1, Net.chain_num + Ab.Num, false);
        bi = new Array(1, Net.chain_num + Ab.Num, true);
        P_unbind_m = new Array(1, bond_max_num + 1, false);
        ln_Hm = new Array(1, bond_max_num + 1, false);
        out_var = new Double[iteration * 2];
    }
}
void SCF_system::z_cross_multi(int& ni, Double& p1, Double& p2, CrossPoint* c, int i2, Array* _t1, Array* _t2, int j)
{
    Chain* ch;
    ch = Net.cross_to_chain[c->id][ni];
    //p2 = ch->Pi->p[i2] * p1;
    if (ch->c1->id != c->id) {
        _t2->aMulti(*ch->Zi012[i2], *_t1);
    }
    else {
        _t2->aMulti(*ch->Zj012[i2], *_t1);
    }
}
void SCF_system::Save_var(char* filename, Array& A) {
    if (A.IsGPU) {
        cpu_temp->aCopy(A);
        SaveData(cpu_temp->p, filename, A.Asize);
    }
    else {
        SaveData(A.p, filename, A.Asize);
    }

}

void SCF_system::Output_data(char* file_name_ex) {
    char filename[50];
    cpu_temp->aCopy(*Net.rho);
    sprintf(filename, "rhoNet_%s.dat", file_name_ex);
    SaveData(cpu_temp->p, filename, Volume);
    cpu_temp->aCopy(*Net.omega);
    sprintf(filename, "omegaNet_%s.dat", file_name_ex);
    SaveData(cpu_temp->p, filename, Volume);
    cpu_temp->aCopy(*Net.chain[0].c1->p);
    sprintf(filename, "P_cross_%s.dat", file_name_ex);
    SaveData(cpu_temp->p, filename, Volume);
    cpu_temp->aCopy(*Ab.rho);
    sprintf(filename, "rhoAb_%s.dat", file_name_ex);
    SaveData(cpu_temp->p, filename, Volume);
    //cpu_temp->aCopy(*Ab.omega);
    //sprintf(filename, "omegaAb_%s.dat", file_name_ex);
    //SaveData(cpu_temp->p, filename, Volume);
    sprintf(filename, "F.dat", file_name_ex);
    SaveData(out_var, filename, 2 * iteration);
}
void SCF_system::Run_an_iter(bool output) {
    int blocks_num_temp, chain_this_batch;
    Double z, t;
    Chain* chain_temp;
    omega_exp1[0].aExp(*(Net.omega), -ds / 2.0);
    omega_exp2[0].aExp(*(Net.omega), -ds / 4.0);
    temp->aExp(*Ab.omega, -1.0);
    h1->aAverage_Around3D(*temp);
    temp->aAdd(*Ab.omega, *NP.U_bond);
    temp->aExp(-1.0);
    h2->aAverage_Around3D(*temp);

    for (int chain_start = 0; chain_start < Net.chain_num; ) {
        chain_this_batch = min(Net.chain_num - chain_start, batch_size);
        for (int j = 0; j < chain_this_batch; j++) {
            q_mid[0]->aCopy(*(Net.chain[j + chain_start].c1->p), Volume, Volume * j * 2, 0);
            q_mid[0]->aCopy(*(Net.chain[j + chain_start].c2->p), Volume, Volume * (j * 2 + 1), 0);
        }

        for (int k = 0; k < Net.s0; k++) {
            DE1.Run_a_step(q_mid, omega_exp1, omega_exp2, k, Net.eta_s);

        }

        blocks_num_temp = (Volume * chain_this_batch + threads - 1) / threads;

        cal_rho_Ab_multi << <blocks_num_temp, threads >> > (q_mid_p_gpu, chain_this_batch, Net.s0, Volume, Volume * chain_this_batch);
        cudaDeviceSynchronize();
        cal_rho_Ab_add << <blocks_num_temp, threads >> > (q_mid_p_gpu, Ab.rho2_p_gpu, chain_start, box_N, Net.s0, Volume, Volume * chain_this_batch);
        cudaDeviceSynchronize();

        for (int j = 0; j < chain_this_batch; j++) {
            q_mid[Net.s0]->aCopy(*q_mid[Net.s0], Volume, H1_start + Volume * j * 2, Volume * j * 2);
            q_mid[Net.s0]->aCopy(*q_mid[Net.s0], Volume, H1_start + Volume * (j * 2 + 1), Volume * (j * 2 + 1));
            q_mid[Net.s0]->aCopy(*q_mid[Net.s0], Volume, H2_start + Volume * j * 2, Volume * j * 2);
            q_mid[Net.s0]->aCopy(*q_mid[Net.s0], Volume, H2_start + Volume * (j * 2 + 1), Volume * (j * 2 + 1));
            //t1->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2);
            //t2->aCopy(*q_mid[Net.s0], Volume, 0, Volume * (j * 2 + 1));
            //t3->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2);
            //t4->aCopy(*q_mid[Net.s0], Volume, 0, Volume * (j * 2 + 1));
            //Save_var("q1.dat", *t1);
            //Save_var("q2.dat", *t2);
            //t1->aMulti(*h1);
            //t2->aMulti(*h1);
            //t3->aMulti(*h2);
            //t4->aMulti(*h2);
            //Save_var("t1.dat", *t1);
            //Save_var("t2.dat", *t2);
            //Save_var("t3.dat", *t3);
            //Save_var("t4.dat", *t4);
            //t1->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2);
            //t2->aCopy(*q_mid[Net.s0], Volume, 0, Volume * (j * 2 + 1));
            //t3->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2);
            //t4->aCopy(*q_mid[Net.s0], Volume, 0, Volume * (j * 2 + 1));
            //t1->aMulti(*h1);
            //t1->aMulti(*t2);
            //cout << "Z1:" << t1->aSum()*DR3 << endl;
            //t3->aMulti(*h2);
            //t3->aMulti(*t4);
            //cout << "Z2:" << t3->aSum()*DR3 << endl;
            q_mid[Net.s0]->aMulti(*h1, Volume, H1_start + Volume * j * 2, 0);
            q_mid[Net.s0]->aMulti(*h1, Volume, H1_start + Volume * (j * 2 + 1), 0);
            q_mid[Net.s0]->aMulti(*h2, Volume, H2_start + Volume * j * 2, 0);
            q_mid[Net.s0]->aMulti(*h2, Volume, H2_start + Volume * (j * 2 + 1), 0);
        }

        for (int k = Net.s0; k < Net.s_N; k++) {
            DE2.Run_a_step(q_mid, omega_exp1, omega_exp2, k, Net.eta_s);
        }
        //temp->aCopy(*q_mid[Net.s_N], Volume, 0, (0 * 2) * Volume + 0);
        //cout << temp->aSum() << endl;
        //temp->aCopy(*q_mid[Net.s_N], Volume, 0, (0 * 2) * Volume + H1_start);
        //cout << temp->aSum() << endl;
        //temp->aCopy(*q_mid[Net.s_N], Volume, 0, (0 * 2) * Volume + H2_start);
        //cout << temp->aSum() << endl;
        //exit(-1);
        //t1->aCopy(*q_mid[Net.s_N], Volume, 0, 0);
        //t2->aCopy(*q_mid[Net.s_N], Volume, 0, Volume);
        //t3->aCopy(*q_mid[Net.s_N], Volume, 0, Volume * 2);
        //t4->aCopy(*q_mid[Net.s_N], Volume, 0, Volume * 3);
        //Save_var("Qi.dat", *t1);
        //Save_var("Qj.dat", *t2);
        //Save_var("Qi1.dat", *t3);
        //Save_var("Qj1.dat", *t4);
        //t3->aCopy(*q_mid[Net.s_N], Volume, 0, Volume * 4);
        //t4->aCopy(*q_mid[Net.s_N], Volume, 0, Volume * 5);
        //Save_var("Qi2.dat", *t3);
        //Save_var("Qj2.dat", *t4);
        //exit(1);
        for (int j = 0; j < chain_this_batch; j++) {
            chain_temp = &Net.chain[j + chain_start];
            chain_temp->Qi->aCopy(*q_mid[Net.s_N], Volume, 0, j * 2 * Volume);
            chain_temp->Qj->aCopy(*q_mid[Net.s_N], Volume, 0, (j * 2 + 1) * Volume);
            //chain_temp->Z1i->aCopy(*q_mid[Net.s_N], Volume, 0, (j * 2) * Volume + H1_start);
            //chain_temp->Z1j->aCopy(*q_mid[Net.s_N], Volume, 0, (j * 2 + 1) * Volume + H1_start);
            //chain_temp->Z2i->aCopy(*q_mid[Net.s_N], Volume, 0, (j * 2) * Volume + H2_start);
            //chain_temp->Z2j->aCopy(*q_mid[Net.s_N], Volume, 0, (j * 2 + 1) * Volume + H2_start);
            chain_temp->c1->p_all[0]->aMulti(*chain_temp->Qi, *chain_temp->c2->p);
            chain_temp->c2->p_all[0]->aMulti(*chain_temp->Qj, *chain_temp->c1->p);
            chain_temp->Qij = chain_temp->c1->p_all[0]->aSum(*temp) * DR3;
            chain_temp->c1->p_all[0]->aMulti(1.0 / chain_temp->Qij);
            chain_temp->c2->p_all[0]->aMulti(1.0 / chain_temp->Qij);
            chain_temp->c1->int_p_h[0] = chain_temp->c1->Cal_H(*chain_temp->c1->p_all[0]);
            chain_temp->c2->int_p_h[0] = chain_temp->c2->Cal_H(*chain_temp->c2->p_all[0]);


            //chain_temp->c1->p_all[1]->aMulti(*chain_temp->Z1i, *chain_temp->c2->p);
            //chain_temp->c2->p_all[1]->aMulti(*chain_temp->Z1j, *chain_temp->c1->p);
            //chain_temp->Z1ij = chain_temp->c1->p_all[1]->aSum(*temp) * DR3;
            //chain_temp->c1->p_all[1]->aMulti(1.0 / chain_temp->Z1ij);
            //chain_temp->c2->p_all[1]->aMulti(1.0 / chain_temp->Z1ij);
            //chain_temp->c1->int_p_h[1] = chain_temp->c1->Cal_H(*chain_temp->c1->p_all[1]);
            //chain_temp->c2->int_p_h[1] = chain_temp->c2->Cal_H(*chain_temp->c2->p_all[1]);

            //chain_temp->c1->p_all[2]->aMulti(*chain_temp->Z2i, *chain_temp->c2->p);
            //chain_temp->c2->p_all[2]->aMulti(*chain_temp->Z2j, *chain_temp->c1->p);
            //chain_temp->Z2ij = chain_temp->c1->p_all[2]->aSum(*temp) * DR3;
            //chain_temp->c1->p_all[2]->aMulti(1.0 / chain_temp->Z2ij);
            //chain_temp->c2->p_all[2]->aMulti(1.0 / chain_temp->Z2ij);
            //chain_temp->c1->int_p_h[2] = chain_temp->c1->Cal_H(*chain_temp->c1->p_all[2]);
            //chain_temp->c2->int_p_h[2] = chain_temp->c2->Cal_H(*chain_temp->c2->p_all[2]);

            chain_temp->Qij = chain_temp->Qij / chain_temp->c1->int_p_h[0] / chain_temp->c2->int_p_h[0];
            chain_temp->Qij = chain_temp->Qij / chain_temp->c1->Z / chain_temp->c2->Z;
            //cout << "Z1*: " << chain_temp->Z1ij << endl;
            //chain_temp->Z1ij = chain_temp->Z1ij / chain_temp->c1->int_p_h[1] / chain_temp->c2->int_p_h[1];
            //chain_temp->Z1ij = chain_temp->Z1ij / chain_temp->c1->Z / chain_temp->c2->Z;
            ////cout << "Z2*: " << chain_temp->Z2ij << endl;
            ////exit(1);
            //chain_temp->Z2ij = chain_temp->Z2ij / chain_temp->c1->int_p_h[2] / chain_temp->c2->int_p_h[2];
            //chain_temp->Z2ij = chain_temp->Z2ij / chain_temp->c1->Z / chain_temp->c2->Z;
        }
        //Save_var("Qi.dat", *Net.chain[0].Qi);
        //Save_var("Z1i.dat", *Net.chain[0].Z1i);
        //Save_var("Z2i.dat", *Net.chain[0].Z2i);
        //exit(0);
        blocks_num_temp = (batch_size * Net.s0 * Volume + threads - 1) / threads;
        //Double* cpu_temp2 = new Double[batch_size * 2 * 3*Volume];
        //int temp_size;
        //for (int i = 0; i < Net.s_N + 1; i++) {
        //    sprintf(string_temp, "./temp_data/q_mid%d.dat", i);
        //    temp_size = q_mid[i]->Asize;
        //    cudaMemcpy(cpu_temp2, q_mid[i]->p, sizeof(Double)* temp_size, cudaMemcpyDeviceToHost);
        //    cudaDeviceSynchronize();
        //    SaveData(cpu_temp2, string_temp, temp_size);
        //}
        //exit(0);
        q_multi_H << <blocks_num_temp, threads >> > (q_mid_p_gpu, batch_size * 2, Volume, Net.s0, Net.s_N, batch_size * Net.s0 * Volume);
        cudaDeviceSynchronize();
        blocks_num_temp = (batch_size * Net.s_N * Volume + threads - 1) / threads;
        q_multi << <blocks_num_temp, threads >> > (q_mid_p_gpu, batch_size, Volume, Net.s_N, batch_size * Net.s_N * Volume);
        cudaDeviceSynchronize();

        q_sum(q_mid_p_gpu, Net.s0, batch_size);

        for (int j = 0; j < chain_this_batch; j++) {
            Net.chain[j + chain_start].rho[0]->aCopy(*q_mid[0], Volume, 0, Volume * j * 2);
            //Net.chain[j + chain_start].rho[1]->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2 + batch_size * 2 * Volume);
            //Net.chain[j + chain_start].rho[2]->aCopy(*q_mid[Net.s0], Volume, 0, Volume * j * 2 + batch_size * 2 * Volume * 2);
        }
        //Save_var("c_rho0.dat", *Net.chain[0].rho[0]);
        //Save_var("c_rho1.dat", *Net.chain[0].rho[1]);
        //Save_var("c_rho2.dat", *Net.chain[0].rho[2]);
        //exit(0);
        if (chain_start == 0) {
            cout << chain_temp->Qij << endl;
            //cout << chain_temp->Z1ij << endl;
            //cout << chain_temp->Z2ij << endl;
        }
        chain_start += chain_this_batch;

    }

    //temp->aAdd(*Ab.omega, *NP.U_bond);
    //temp->aExp(-1.0);
    //for (int j = 0; j < Net.chain_num; j++) {
    //    Ab.rho3[j]->aCopy(*Ab.rho2[j]);
    //    Ab.rho3[j]->aMulti(*temp);
    //    t = Ab.rho3[j]->aSum(*t1) * DR3;
    //    Ab.rho3[j]->aMulti(1.0 / t);
    //    //cout << "Ab_rho3:" << Ab.rho3[j]->aSum(*t1) * DR3 << endl;
    //}
    //temp->aExp(*Ab.omega, -1.0);
    //for (int j = 0; j < Net.chain_num; j++) {
    //    Ab.rho2[j]->aMulti(*temp);
    //    t = Ab.rho2[j]->aSum(*t1) * DR3;
    //    Ab.rho2[j]->aMulti(1.0 / t);
    //    //cout << "Ab_rho2:" << Ab.rho2[j]->aSum(*t1) * DR3 << endl;
    //}
    //exit(-1);
    //for (int j = 0; j < Net.chain_num; j++) {
    //    dG_cnf1->p[j] = -log(Net.chain[j].Z1ij) + log(Net.chain[j].Qij) + log(Ab.Z0);
    //    dG_cnf2->p[j] = -log(Net.chain[j].Z2ij) + log(Net.chain[j].Qij) + log(Ab.Zb);
    //    if (IsUnbind) {
    //        dG_cnf2->p[j] += 200.0;
    //    }
    //    if (j == 0) {
    //        cout << "dG_cnf:" << dG_cnf1->p[j] << ", " << dG_cnf2->p[j] << endl;
    //    }
    //}
    //int m_shape[] = { 4,NANO_M + 1 };
    //Array zi(2, m_shape, false);
    //for (int m = 0; m < bond_max_num + 1; m++) {
    //    z = m * log(Ab.Zb) + (Ab.Num - m) * log(Ab.Z0) - m * dG_NP;
    //    cout << "Z_Ab: " << Ab.Z0 / Ab.Zb << endl;
    //    zi.p[m] = z;
    //    dGij->Init(dG_Net);
    //    for (int j = 0; j < Net.chain_num; j++) {
    //        if (m > 0) {
    //            temp->SetValue(dG_cnf2->p[j], m, 0);
    //        }
    //        if (m < Ab.Num) {
    //            temp->SetValue(dG_cnf1->p[j], Ab.Num - m, m);
    //        }
    //        dGij->aAdd(*temp, Ab.Num, j * Ab.Num, 0);
    //        z += log(Net.chain[j].Qij);
    //    }
    //    zi.p[m + NANO_M + 1] = z - zi.p[m];
    //    dGij->aExp(-1.0);
    //    cout << "dGij:" << dGij->aSum(*temp) << "; m:" << m << endl;
    //    //cpu_temp->aCopy(*dGij);
    //    //SaveData(cpu_temp->p, "dGij.dat", Net.chain_num * Ab.Num);
    //    Cal_link_p(dGij, bij, bij_temp, bi, Net.chain_num, Ab.Num, bij_iter);
    //    cpu_temp->aCopy(*bij);
    //    sprintf(string_temp, "bij_%d.dat", m);
    //    SaveData(cpu_temp->p, string_temp, Net.chain_num * Ab.Num);
    //    cpu_temp->aCopy(*bi);
    //    sprintf(string_temp, "bi_%d.dat", m);
    //    SaveData(cpu_temp->p, string_temp, Net.chain_num + Ab.Num);
    //    //exit(-1);
    //    P_unbind_m->p[m] = 1.0;
    //    bi_cpu->aCopy(*bi);
    //    for (int j = 0; j < m; j++) {
    //        P_unbind_m->p[m] *= bi_cpu->p[j + Net.chain_num];
    //    }
    //    bi->aLn(1.0);
    //    bi_cpu->aLn(1.0);
    //    cout << "bi_sum:" << bi_cpu->aSum() << "; m:" << m << endl;
    //    cout << "bij_sum:" << bij->aSum(*bij_temp) << "; m:" << m << endl;
    //    cout << "P_unbind_m:" << P_unbind_m->p[m] << "; m:" << m << endl;
    //    ln_Hm->p[m] = z - bi->aSum() - bij->aSum(*bij_temp);
    //    zi.p[m + (NANO_M + 1) * 2] = ln_Hm->p[m] - z;
    //    cout << "ln_Hm:" << ln_Hm->p[m] << "; m:" << m << endl;
    //    ln_Hm->p[m] = ln_Hm->p[m] + log_fact(Ab.Num) + log_fact(NP.M) - log_fact(m) - log_fact(Ab.Num - m) - log_fact(NP.M - m);
    //    zi.p[m + (NANO_M + 1) * 3] = log_fact(Ab.Num) + log_fact(NP.M) - log_fact(m) - log_fact(Ab.Num - m) - log_fact(NP.M - m);
    //    //exit(-1);
    //    for (int j = 0; j < Net.chain_num; j++) {
    //        if (m == 0) {
    //            temp->aCopy(*bij, Ab.Num, 0, j * Ab.Num);
    //            Net.chain[j].p_i_m[1]->p[m] = temp->aSum(Ab.Num);
    //            Net.chain[j].p_i_m[2]->p[m] = 0.0;
    //            Net.chain[j].p_i_m[0]->p[m] = 1.0 - Net.chain[j].p_i_m[1]->p[m] - Net.chain[j].p_i_m[2]->p[m];
    //        }
    //        else {
    //            temp->aCopy(*bij, m, 0, j * Ab.Num);
    //            Net.chain[j].p_i_m[2]->p[m] = temp->aSum(m);
    //            if (m < Ab.Num) {
    //                temp->aCopy(*bij, Ab.Num - m, 0, j * Ab.Num + m);
    //                Net.chain[j].p_i_m[1]->p[m] = temp->aSum(Ab.Num - m);
    //            }
    //            else {
    //                Net.chain[j].p_i_m[1]->p[m] = 0.0;
    //            }
    //            Net.chain[j].p_i_m[0]->p[m] = 1.0 - Net.chain[j].p_i_m[1]->p[m] - Net.chain[j].p_i_m[2]->p[m];
    //        }
    //    }
    //}
    //Double ln_H0;
    //ln_H0 = ln_Hm->p[0];
    //for (int m = 0; m < bond_max_num + 1; m++) {
    //    if (ln_H0 < ln_Hm->p[m]) {
    //        ln_H0 = ln_Hm->p[m];
    //    }
    //}
    //ln_Hm->aAdd(-ln_H0);
    //ln_Hm->aExp(1.0);
    //t = ln_Hm->aSum(*temp);
    //cout << "ln_HmS0:" << t << endl;
    //ln_Z = -log_fact(NP.M) - log_fact(Ab.Num) + log(t) + ln_H0;
    //F = -ln_Z;
    //ln_Hm->aMulti(1.0 / t);
    //Save_var("P_m.dat", *ln_Hm);
    //Save_var("z_m.dat", zi);
    //Save_var("P_ubind_m.dat", *P_unbind_m);
    //exit(1);
    //for (int j = 0; j < Net.chain_num; j++) {
    //    Net.chain[j].Pi->p[1] = 0.0;
    //    Net.chain[j].Pi->p[2] = 0.0;
    //    for (int m = 0; m < bond_max_num + 1; m++) {
    //        Net.chain[j].Pi->p[1] += (Net.chain[j].p_i_m[1]->p[m] * ln_Hm->p[m]);
    //        Net.chain[j].Pi->p[2] += (Net.chain[j].p_i_m[2]->p[m] * ln_Hm->p[m]);
    //    }
    //    Net.chain[j].Pi->p[0] = 1.0 - Net.chain[j].Pi->p[1] - Net.chain[j].Pi->p[2];
    //}
    //cout << "ln_HmS:" << ln_Hm->aSum(*temp) << endl;
    //t = 0.;
    //P_unbind = 0.;
    //for (int m = 0; m < bond_max_num + 1; m++) {
    //    t += m * ln_Hm->p[m];
    //    P_unbind += P_unbind_m->p[m] * ln_Hm->p[m];
    //}
    //Ab.n1 = t;
    //Ab.n0 = Ab.Num - t;

    //for (int j = 0; j < Net.chain_num; j++) {
    //    Ab.n1 -= Net.chain[j].Pi->p[2];
    //    Ab.n0 -= Net.chain[j].Pi->p[1];
    //}
    //cout << "Ab:" << Ab.n0 << ";" << Ab.n1 << endl;
    Net.rho->SetValue(0.0);
    //Save_var("Ab0.dat", *Ab.rho0);
    //Save_var("Ab1.dat", *Ab.rho1);
    //Ab.rho->aWAdd(*Ab.rho0, *Ab.rho1, Ab.n0, Ab.n1);
    Double _z = 0.0;
    for (int j = 0; j < Net.chain_num; j++) {
        _z += log(Net.chain[j].Qij);
    }
    F = -_z;
    for (int j = 0; j < Net.chain_num; j++) {
        //if (j == 0) {
        //    int _j = 0;
        //    Save_var("chain_rho0.dat", *Net.chain[_j].rho[0]);
        //    Save_var("chain_rho1.dat", *Net.chain[_j].rho[1]);
        //    Save_var("chain_rho2.dat", *Net.chain[_j].rho[2]);
        //    cout << "chain_pi" << Net.chain[_j].Pi->p[0] << ";" << Net.chain[_j].Pi->p[1];
        //    cout << ";" << Net.chain[_j].Pi->p[2] << endl;
        //    exit(1);
        //}
        /*for (int k = 0; k < 3; k++) {
            Net.chain[j].rho[k]->aMulti(Net.chain[j].Pi->p[k]);
            
        }*/
        Net.rho->aAdd(*Net.chain[j].rho[0]);
        //Ab.rho2[j]->aMulti(Net.chain[j].Pi->p[1]);
        //Ab.rho3[j]->aMulti(Net.chain[j].Pi->p[2]);
        //Ab.rho->aAdd(*Ab.rho2[j]);
        //Ab.rho->aAdd(*Ab.rho3[j]);
    }
    //cout << "Ab_rho:" << Ab.rho->aSum(*temp) << endl;
    Double p1, p2, p3, p4, p5;
    CrossPoint* c;
    Chain* ch;
    int n1, n2, n3, n4, n5;
    int debug_i = 0;
    int debug_cid = -1;
    int Isout = 0;
    Array debug_p(1, 18, false);
    int debug_p_i = 0;
    for (int j = 0; j < Net.chain_num; j++) {
        if (Isout>=1) {
            break;
        }
        for (int ic = 0; ic < 2; ic++) {
            if (Isout>=1) {
                break;
            }
            if (ic == 0) {
                c = Net.chain[j].c1;
            }
            else {
                c = Net.chain[j].c2;
            }
            if (!c->Isfree) {
                continue;
            }
            else {
                if (debug_cid == -1) {
                    debug_cid = c->id;
                }
                if (debug_cid==c->id){
                    Isout++;
                    for (int _i = 0; _i < 6; _i++) {
                        ch = Net.cross_to_chain[c->id][_i];
                        for (int _j = 0; _j < 1; _j++) {
                            if (ch->c1->id != c->id) {
                                sprintf(string_temp, "Z_chain%d_s%d.dat", _i, _j);
                                Save_var(string_temp, *ch->Zi012[_j]);
                            }
                            else {
                                sprintf(string_temp, "Z_chain%d_s%d.dat", _i, _j);
                                Save_var(string_temp, *ch->Zj012[_j]);
                            }
                            debug_p.p[debug_p_i]= ch->Pi->p[_j];
                            debug_p_i++;
                        }
                    }
                }
                
            }
        }
    }
    sprintf(string_temp, "chain_state_cid%d.dat", debug_cid);
    Save_var(string_temp, debug_p);
    for (int j = 0; j < Net.chain_num; j++) {
        for (int ic = 0; ic < 2; ic++) {
            if (ic == 0) {
                c = Net.chain[j].c1;
            }
            else {
                c = Net.chain[j].c2;
            }
            if (!c->Isfree) {
                continue;
            }
            c->p_n->SetValue(0.0);
            n1 = 0;
            for (int i1 = 0; i1 < 1; i1++) {
                if (Net.cross_to_chain[c->id][n1]->id == Net.chain[j].id) {
                    n1++;
                }
                
                ch = Net.cross_to_chain[c->id][n1];
                n2 = n1 + 1;
                //p1 = ch->Pi->p[i1];
                if (ch->c1->id != c->id) {
                    t1->aCopy(*ch->Zi012[i1]);
                }
                else {
                    t1->aCopy(*ch->Zj012[i1]);
                }
                for (int i2 = 0; i2 < 1; i2++) {
                    if (Net.cross_to_chain[c->id][n2]->id == Net.chain[j].id) {
                        n2++;
                    }
                    z_cross_multi(n2, p1, p2, c, i2, t1, t2, j);
                    n3 = n2 + 1;
                    for (int i3 = 0; i3 < 1; i3++) {
                        if (Net.cross_to_chain[c->id][n3]->id == Net.chain[j].id) {
                            n3++;
                        }
                        z_cross_multi(n3, p2, p3, c, i3, t2, t3, j);
                        n4 = n3 + 1;
                        for (int i4 = 0; i4 < 1; i4++) {
                            if (Net.cross_to_chain[c->id][n4]->id == Net.chain[j].id) {
                                n4++;
                            }
                            z_cross_multi(n4, p3, p4, c, i4, t3, t4, j);
                            n5 = n4 + 1;
                            for (int i5 = 0; i5 < 1; i5++) {
                                if (Net.cross_to_chain[c->id][n5]->id == Net.chain[j].id) {
                                    n5++;
                                }
                                z_cross_multi(n5, p4, p5, c, i5, t4, t5, j);
                                //t5->aMulti(p5);
                                c->p_n->aAdd(*t5);

                            }

                        }

                    }

                }

            }
            t = c->p_n->aSum(*temp) * DR3;
            cout << "cross-t:" << t << endl;
            c->p_n->aMulti(1.0 / t);

        }
    }

    for (int j = 0; j < Net.chain_num; j++) {
        chain_temp = &Net.chain[j];
        if (chain_temp->c1->Isfree) {
            //cpu_temp->aCopy(*chain_temp->c1->p);
            //SaveData(cpu_temp->p, "p_0.dat", Volume);

            chain_temp->c1->p_n->Positive_Ensure();
            //cpu_temp->aCopy(*chain_temp->c1->p_n);
            //SaveData(cpu_temp->p, "p_n.dat", Volume);
            chain_temp->c1->h_n->aLn(*chain_temp->c1->p_n, -1.0);

            cout << "cross-hn:" << chain_temp->c1->h_n->aSum(*temp) << endl;
            //cpu_temp->aCopy(*chain_temp->c1->h_n);
            //SaveData(cpu_temp->p, "h_n.dat", Volume);
            //cpu_temp->aCopy(*chain_temp->c1->h);
            //SaveData(cpu_temp->p, "h_0.dat", Volume);
            chain_temp->c1->h->aWAdd(*chain_temp->c1->h_n, 1.0 - step, step);
            //cpu_temp->aCopy(*chain_temp->c1->h);
            //SaveData(cpu_temp->p, "h_n0.dat", Volume);
            chain_temp->c1->Cal_Z();
            chain_temp->c1->p->aExp(*chain_temp->c1->h, -1.0);
            cout << "cross-p:" << chain_temp->c1->p->aSum(*temp) << endl;
            chain_temp->c1->p->aMulti(1.0 / chain_temp->c1->Z);
            cpu_temp->aCopy(*chain_temp->c1->p);
            sprintf(string_temp, "./p_chain%d_cid%d_it%d.dat", j, chain_temp->c1->id, it);
            if ((it+1) % 10 == 0) {
                SaveData(cpu_temp->p, string_temp, Volume);
            }
            

        }
        if (chain_temp->c2->Isfree) {
            chain_temp->c2->p_n->Positive_Ensure();
            chain_temp->c2->h_n->aLn(*chain_temp->c2->p_n, -1.0);
            chain_temp->c2->h->aWAdd(*chain_temp->c2->h_n, 1.0 - step, step);
            chain_temp->c2->Cal_Z();
            chain_temp->c2->p->aExp(*chain_temp->c2->h, -1.0);
            chain_temp->c2->p->aMulti(1.0 / chain_temp->c2->Z);
            cpu_temp->aCopy(*chain_temp->c2->p);
            sprintf(string_temp, "./p_chain%d_cid%d_it%d.dat", j, chain_temp->c2->id, it);
            if ((it + 1) % 10 == 0) {
                SaveData(cpu_temp->p, string_temp, Volume);
            }
        }
    }
    t = Net.rho->aSum(*temp) * DR3;
    cout << "net_rho: " << t << endl;
    Net.rho->aMulti(Net.chain_num * Net.s_N * DS / t);
    //t = Ab.rho->aSum(*temp) * DR3;
    //Ab.rho->aMulti(Ab.Num / t);
    Net.omega_n->aCopy(*Net.rho);
    //Net.omega_n->aWAdd(*NP.WCA, v0, 1.0);
    Net.omega->aWAdd(*Net.omega_n, 1.0 - step, step);

    //Ab.Cal_Z(NP);
    out_var[it * 2] = F;
    out_var[it * 2 + 1] = 0.0;
    it++;
    if (it >= 29) {
        step = 0.1;
    }
}
void CrossPoint::h2p() {
    p->aExp(*h, -1.0);
}
void CrossPoint::Update_h(Array& n_p, Double step) {
    temp->aLn(n_p, -1.0);
    h->aWAdd(*temp, 1.0 - step, step);
}
Double CrossPoint::Cal_H(Array& n_p) {
    temp->aMulti(n_p, *h);

    return temp->aSum(h->Asize) * DR3;
}
Double CrossPoint::Cal_Z() {
    temp->aExp(*h, -1.0);
    Z = temp->aSum(h->Asize) * DR3;

    return Z;
}
//void Cal_q(cufftHandle planf, cufftHandle planb, Double** q_mid,  Double**omega1,Double**omega2, int* eta_s,Double* k_fft, Double* k2_fft, CufftComplex* qf,Double* q_temp, int it_num,bool IsTwice)
//{
//    Double k_inverse = 1.0 / V;
//    int blocks_n= number_of_blocks_n;
//    int vm=Vm;
//    int start = 0;
//    if (IsTwice) {
//        blocks_n = number_of_blocks_n * 3;
//        vm = Vm * 3;
//        start = eta_num_s0;
//    }
//    for (int i = start; i < start+it_num; i++)
//    {
//        //Cuda_err_print();
//        MultiD2D_2d << <blocks_n, threads_per_block >> > (omega1[eta_s[i]], q_mid[i], q_mid[i+1],vm,V);
//        //Cuda_err_print();
//        CufftExcForward(planf, q_mid[i + 1], qf);
//
//        MultiToD2Z << <blocks_n, threads_per_block >> > (k_fft, qf, vm, V);
//        CufftExcBackward(planb, qf, q_mid[i + 1]);
//        MultiTo << <blocks_n, threads_per_block >> > (q_mid[i + 1], k_inverse, vm);
//        MultiToD2D_2d << <blocks_n, threads_per_block >> > (omega1[eta_s[i]], q_mid[i + 1], vm, V);
//        Copy << <blocks_n, threads_per_block >> > (q_mid[i], q_temp, vm);
//        for (int j = 0; j < 2; j++)
//        {
//            MultiToD2D_2d << <blocks_n, threads_per_block >> > (omega2[eta_s[i]], q_temp, vm, V);
//            CufftExcForward(planf, q_temp, qf);
//            MultiToD2Z << <blocks_n, threads_per_block >> > (k2_fft, qf, vm, V);
//            CufftExcBackward(planb, qf, q_temp);
//            MultiTo << <blocks_n, threads_per_block >> > (q_temp, k_inverse, vm);
//            MultiToD2D_2d << <blocks_n, threads_per_block >> > (omega2[eta_s[i]], q_temp, vm, V);
//        }
//        q_mean_al << <blocks_n, threads_per_block >> > (q_temp, q_mid[i + 1], vm);
//        q_mean << <blocks_n, threads_per_block >> > (q_mid[i + 1], q_mid[i], vm);
//    }
//
//}
//
//void Sum(Double* a, int N_array)
//{
//    N_array = N_array / 2;
//    Double zero = 0.0;
//    int threads_per_block_s;
//    int number_of_blocks_s;
//    for (N_array; N_array > 0; N_array = N_array / 2)
//    {
//        threads_per_block_s = min(N_array, 512);
//        number_of_blocks_s = (N_array + threads_per_block_s - 1) / threads_per_block_s;
//        Sum_gpu << <number_of_blocks_s, threads_per_block_s >> > (a, N_array);
//        //Cuda_err_print();
//        if (N_array>1&&N_array % 2 == 1)
//        {
//            cudaMemcpy(&a[N_array], &zero, sizeof(Double), cudaMemcpyHostToDevice);
//            //Cuda_err_print();
//            N_array += 1;
//        }
//    }
//
//}
//void q_sum_block(Double* q_start, int length)
//{
//    int blocks_num;
//    bool length_is_odd = (length % 2 == 1);
//    int len;
//    int N;
//    if (length_is_odd)
//    {
//        len = (length - 1) / 2;
//    }
//    else
//    {
//        len = (length) / 2;
//    }
//    for (len; len > 0; len = len / 2)
//    {
//        N = len * V;
//        blocks_num = (N + threads_per_block - 1) / threads_per_block;
//        q_sum << <blocks_num, threads_per_block >> > (q_start, V, len, N);
//        if (length_is_odd) {
//            Copy << <number_of_blocks, threads_per_block >> > (&q_start[(length - 1) * V], &q_start[len * V], V);
//            length_is_odd = false;
//            len += 1;
//        }
//        if (len > 1 && len % 2 == 1)
//        {   
//            set_value << <number_of_blocks, threads_per_block >> > (q_start, 0.0, V * len, V);
//            len += 1;
//        }
//    }
//}

//void Cal_rho(Double** q_mid,Double** q_mid_p_gpu, int c1,int c2,Double** rho,int* block, Double Q1, Double* q_sum_temp)
//{
//    int N = eta_num * V;
//    //Double test = 0.0;
//    q_multi << <number_of_blocks_sum, threads_per_block >> > (q_mid_p_gpu, q_sum_temp, c1, c2, V, eta_num, N);
//    for (int i = 0; i < block_num; i++)
//    {
//        q_sum_block(&q_sum_temp[block[i] * V], block[i+1]-block[i]);
//        MultiTo << <number_of_blocks, threads_per_block >> > (&q_sum_temp[block[i] * V], ds / Q1,V);
//        AddTo << <number_of_blocks, threads_per_block >> > (rho[block_type[i]], &q_sum_temp[block[i] * V], V);
//    }
//}
//Double Cal_Ab(Double**rho,Double**omega_Ab,Double set_bond, Double Ab_free,bool bond_opt,Double delta_G,Double* Z) //return new bond num
//{
//    Copy << <number_of_blocks, threads_per_block >> > (omega_Ab[0], rho[rAbb_id],V);
//    Sum(omega_Ab[0], V);
//    cudaMemcpy(&Z[0], &omega_Ab[0][0], sizeof(Double), cudaMemcpyDeviceToHost);
//    //Cuda_err_print();
//    Copy << <number_of_blocks, threads_per_block >> > (omega_Ab[1], rho[rAbf_id], V);
//    cudaMemcpy(&Z[0], &omega_Ab[0][0], sizeof(Double), cudaMemcpyDeviceToHost);
//    //Cuda_err_print();
//
//    Sum(omega_Ab[1], V);
//    //Cuda_err_print();
//    cudaMemcpy(&Z[1], &omega_Ab[1][0], sizeof(Double), cudaMemcpyDeviceToHost);
//    Z[0] *= dr3;
//    Z[1] *= dr3;
//    MultiTo << <number_of_blocks, threads_per_block >> > (rho[rAbb_id], set_bond / Z[0], V);
//    MultiTo << <number_of_blocks, threads_per_block >> > (rho[rAbf_id], Ab_free / Z[1], V);
//
//    if (bond_opt)
//    {
//        Double N_Ab = set_bond + Ab_free;
//        Double K;
//        K = Z[1] / Z[0] * exp(delta_G) + M + N_Ab;
//        return (K - sqrt(K * K - 4 * M * N_Ab)) / 2.0;
//    }
//    else
//    {
//        return set_bond;
//    }
//}
//
//void Cal_F(Double* F, Double** rho,Double** rho_p_gpu,Double**omega,Double** omega_p_gpu, Double*Z,Double*Q,Double*WCA,Double*u,Double*eps,Double set_bond, Double Ab_free, bool bond_opt, Double delta_G,int it)
//{
//    //F:enthapy,bond,entropyAb,entropyNet,con,react,Total
//    Double* temp;
//    cudaMalloc((void**)&temp, size_DOUBLE);
//    Cal_fe << <number_of_blocks, threads_per_block >> > (rho_p_gpu, temp,eps,WCA,V);
//    Sum(temp, V);
//    cudaMemcpy(&F[it * F_num], &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    F[it * F_num] = F[it * F_num] * dr3;
//
//    Multi << <number_of_blocks, threads_per_block >> > (u, rho[2], temp, V);
//    Sum(temp, V);
//    cudaMemcpy(&F[it * F_num+1], &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    F[it * F_num + 1] = F[it * F_num + 1] * dr3;
//
//    Cal_fsAb << <number_of_blocks, threads_per_block >> > (rho_p_gpu, omega_p_gpu, temp, V);
//    Sum(temp, V);
//    cudaMemcpy(&F[it * F_num+2], &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    F[it * F_num + 2] = F[it * F_num + 2] * dr3 - Ab_free * log(Z[1]) - set_bond * log(Z[0]);
//    Cal_fsNet << <number_of_blocks, threads_per_block >> > (rho_p_gpu, omega_p_gpu, temp, V);
//    Sum(temp, V);
//    cudaMemcpy(&F[it * F_num+3], &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    F[it * F_num + 3] = F[it * F_num + 3] * dr3;
//    for (int i = 0; i < chain_num; i++) {
//        F[it * F_num + 3] -= log(Q[i]);
//    }
//    if (bond_opt) {
//        F[it * F_num + 4] = set_bond * log(set_bond) + Ab_free * log(Ab_free) + (M - set_bond) * log(M - set_bond) - set_bond - Ab_free - (M - set_bond);
//    }
//    else {
//        F[it * F_num + 4] = log_fact(lround(set_bond)) + log_fact(lround(Ab_free)) + log_fact(lround(M - set_bond));
//    }
//    F[it * F_num + 5] = set_bond * delta_G;
//    F[it * F_num + 6] = F[it * F_num] + F[it * F_num + 1] + F[it * F_num + 2] + F[it * F_num + 3] + F[it * F_num + 4] + F[it * F_num + 5];
//    cudaFree(temp);
//}
//
//void Normalize_rho(Double** rho) {
//    Double* temp;
//    Double s;
//    cudaMalloc((void**)&temp, size_DOUBLE);
//    Copy << <number_of_blocks, threads_per_block >> > (rho[rA_id], temp, V);
//    Sum(temp, V);
//    cudaMemcpy(&s, &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    s = 0.8 * chain_num * chain_l / dr3 / s;
//    MultiTo << <number_of_blocks, threads_per_block >> > (rho[rA_id], s, V);
//    Copy << <number_of_blocks, threads_per_block >> > (rho[rB_id], temp, V);
//    Sum(temp, V);
//    cudaMemcpy(&s, &temp[0], sizeof(Double), cudaMemcpyDeviceToHost);
//    s = 0.2 * chain_num * chain_l / dr3 / s;
//    MultiTo << <number_of_blocks, threads_per_block >> > (rho[rB_id], s, V);
//    cudaFree(temp);
//}
//void Constraint_MA(Double**rho, Double set_MA, Double punish_lam, Double punish_c,Double** omega_ex, Double ma) {
//    //omega_ex: B,Ab
//    Double temp;
//    temp = punish_c * (ma - set_MA)  + punish_lam;
//    MultiK2D << <number_of_blocks, threads_per_block >> > (rho[2], temp, omega_ex[0], V);
//    temp = punish_c * (ma - set_MA) + punish_lam;
//    MultiK2D << <number_of_blocks, threads_per_block >> > (rho[1], temp, omega_ex[1], V);
//}

//void Data_init(Double* k_fft, Double* k2_fft, Double** omega, int* crosslink, int* eta_s, int* chains, int* batch_cross, int* batch_chain, Double* WCA, Double* u) {
//    ifstream ifs("../k_fft.dat", std::ios::binary | std::ios::in);
//    ifs.read((char*)k_fft, sizeof(Double) * V);
//    ifs.close();
//    ifstream ifs2("../k2_fft.dat", std::ios::binary | std::ios::in);
//    ifs2.read((char*)k2_fft, sizeof(Double) * V);
//    ifs2.close();
//    ifstream ifs3("omega_A_init.dat", std::ios::binary | std::ios::in);
//    ifs3.read((char*)omega[0], sizeof(Double) * V);
//    ifs3.close();
//    ifstream ifs4("omega_B_init.dat", std::ios::binary | std::ios::in);
//    ifs4.read((char*)omega[1], sizeof(Double) * V);
//    ifs4.close();
//    ifstream ifs5("omega_Ab_bond_init.dat", std::ios::binary | std::ios::in);
//    ifs5.read((char*)omega[2], sizeof(Double) * V);
//    ifs5.close();
//    ifstream ifs6("omega_Ab_free_init.dat", std::ios::binary | std::ios::in);
//    ifs6.read((char*)omega[3], sizeof(Double) * V);
//    ifs6.close();
//    ifstream ifs7("../crosslink.dat", std::ios::binary | std::ios::in);
//    ifs7.read((char*)crosslink, sizeof(int) * crosslink_num);
//    ifs7.close();
//    ifstream ifs8("../eta_s.dat", std::ios::binary | std::ios::in);
//    ifs8.read((char*)eta_s, sizeof(int) * eta_num);
//    ifs8.close();
//    ifstream ifs9("../chains.dat", std::ios::binary | std::ios::in);
//    ifs9.read((char*)chains, sizeof(int) * chain_num * 2);
//    ifs9.close();
//    ifstream ifs10("../batch_chain.dat", std::ios::binary | std::ios::in);
//    ifs10.read((char*)batch_chain, sizeof(int) * chain_total);
//    ifs10.close();
//    ifstream ifs11("../batch_cross.dat", std::ios::binary | std::ios::in);
//    ifs11.read((char*)batch_cross, sizeof(int) * cross_total);
//    ifs11.close();
//    ifstream ifs12("WCA.dat", std::ios::binary | std::ios::in);
//    ifs12.read((char*)WCA, sizeof(Double) * V);
//    ifs12.close();
//    ifstream ifs13("u.dat", std::ios::binary | std::ios::in);
//    ifs13.read((char*)u, sizeof(Double) * V);
//    ifs13.close();
//    return;
//
//}
void New_cpu(Double** a, int n, int size)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = (Double*)malloc(size);
    }
}
void Free_cpu(Double** a, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(a[i]);
    }
}

int main(int argc, char* argv[])
{
    //argv:1-N_Ab,2-set_bond,3-set_MA,4-delta_G,5-step,6-iteration
    //7-lam_step,8-punish_lam,9-p*_c,10-p*_rho
    ///cpu var

    bool IsUnbind = true;
    Double N_Ab = strtod(argv[1], NULL);//100
    Double set_bond0 = strtod(argv[2], NULL);//1.0
    Double dG_NP = strtod(argv[3], NULL); //-1.0
    Double dG_Net = strtod(argv[4], NULL); //-1.0
    Double step = strtod(argv[5], NULL); //0.5
    int iteration = atoi(argv[6]); //20
    int bij_iteration = atoi(argv[7]); //50
    if (set_bond0 < -0.1) {
        IsUnbind = false;
    }
    SCF_system scf;
    scf.dG_NP = dG_NP;
    scf.dG_Net = dG_Net;
    scf.batch_size = BATCH_SIZE;
    scf.box_N = BOX_N;
    scf.Volume = VOLUME;
    scf.bij_iter = bij_iteration;
    scf.IsUnbind = IsUnbind;
    scf.ds = DS;
    scf.v0 = V0;
    scf.step = step;
    scf.Ab.Num = N_Ab;
    scf.Net.chain_num = CHAIN_NUM;
    scf.Net.cross_num = CROSSLINK_NUM;
    scf.Net.s0 = ETA_NUM_S0;
    scf.Net.s_N = ETA_NUM;
    scf.NP.M = RECEPTOR_NUM;
    scf.iteration = iteration;

    //int shape[] = { 81,100 };
    //Array bij_temp(2, shape, true);
    //Array bij_cpu(2, shape, false);
    //bij_temp.SetValue(1.0);
    //bij_temp.Sum2D(1);
    //bij_cpu.aCopy(bij_temp);

    scf.Init();
    char filename_ex[50];

    //Cuda_err_print(0);
    Cuda_err_print(0);

    for (int it = 0; it < iteration; it++) {
        scf.Run_an_iter();
        sprintf(filename_ex, "_%d", it);
        scf.Output_data(filename_ex);
    }
    return 0;
}
