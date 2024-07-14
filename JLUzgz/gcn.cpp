#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

//#define USEOPENMP

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname) {
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  // raw_graph.resize(e_num * 2);

  while (!infile.eof()) {
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

void raw_graph_to_AdjacencyList() {
  int src;
  int dst;

  edge_index.resize(v_num);
  edge_val.resize(v_num);
  degree.resize(v_num, 0);

#ifdef USEOPENMP
#pragma omp parallel for 
#endif
  for (int i = 0; i < raw_graph.size() / 2; i++) {
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    edge_index[dst].push_back(src);
    degree[src]++;
  }
}

void edgeNormalization() {
#ifdef USEOPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < edge_index[i].size(); j++) {
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
      edge_val[i].push_back(val);
    }
  }
}

void readFloat(char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
  float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < out_dim; j++) {
      for (int k = 0; k < in_dim; k++) {
        tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
      }
    }
  }
}

void AX(int dim, float *in_X, float *out_X) {
  float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
  float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

#ifdef USEOPENMP
#pragma omp parallel for collaspe(2)
#endif
  for (int i = 0; i < v_num; i++) {
    vector<int> &nlist = edge_index[i];
    for (int j = 0; j < nlist.size(); j++) {
      int nbr = nlist[j];
      for (int k = 0; k < dim; k++) {
        tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
      }
    }
  }
}

void ReLU(int dim, float *X) {
#ifdef USEOPENMP
#pragma omp parallel for
#endif 
  for (int i = 0; i < v_num * dim; i++)
    if (X[i] < 0) X[i] = 0;
}

void LogSoftmax(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

#ifdef USEOPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < v_num; i++) {
    float max = tmp_X[i][0];
    for (int j = 1; j < dim; j++) {
      if (tmp_X[i][j] > max) max = tmp_X[i][j];
    }

    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += exp(tmp_X[i][j] - max);
    }
    sum = log(sum);

    for (int j = 0; j < dim; j++) {
      tmp_X[i][j] = tmp_X[i][j] - max - sum;
    }
  }
}

float MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;
  float max = -__FLT_MAX__;

#ifdef USEOPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < v_num; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += tmp_X[i][j];
    }
    if (sum > max) max = sum;
  }
  return max;
}

void freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

void somePreprocessing() {
  // The graph  will be transformed into adjacency list, you can use other data
  // structure such as CSR
  raw_graph_to_AdjacencyList();
}

static void mm_generate(float* matA,float* matB,float* matC,const int M,const int N,const int K,const int strideA,const int strideB,const int strideC)
{
#ifdef USEOPENMP
#pragma omp parallel for collaspe(2)
#endif
	for (int i = 0; i < M;i++)
	{
		for (int j = 0; j < N;j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K;k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] = sum;
		}
	}
}

static void mm_winograd(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0))
	{
		return mm_generate(matA, matB, matC, M, N, K, strideA, strideB, strideC);
	}
	memset(matC, 0, M*strideC*sizeof(float));
	int offset = 0;

	std::vector<float> S1((M / 2) * (K / 2));
	std::vector<float> S2((M / 2) * (K / 2));
	std::vector<float> S3((M / 2) * (K / 2));
	std::vector<float> S4((M / 2) * (K / 2));
	for (int i = 0; i < M / 2;i++)
	{
		for (int j = 0; j < K / 2;j++)
		{
			const int idx = i*K / 2 + j;
			//S1 = A21 + A22
			S1[idx] = matA[(i + M / 2)*strideA + j] + matA[(i + M / 2)*strideA + j + K / 2];
			//S2 = S1 - A11
			S2[idx] = S1[idx] - matA[i*strideA + j];
			//S3 = A11 - A21
			S3[idx] = matA[i*strideA + j] - matA[(i + M / 2)*strideA + j];
			//S4 = A12 - S2
			S4[idx] = matA[i*strideA + j + K / 2] - S2[idx];
		}
	}
	std::vector<float> T1((K / 2) * (N / 2));
	std::vector<float> T2((K / 2) * (N / 2));
	std::vector<float> T3((K / 2) * (N / 2));	
	std::vector<float> T4((K / 2) * (N / 2));
	for (int i = 0; i < K / 2; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			const int idx = i*N / 2 + j;
			//T1 = B21 - B11
			T1[idx] = matB[(i + K / 2)*strideB + j] - matB[i*strideB + j];
			//T2 = B22 - T1
			T2[idx] = matB[(i + K / 2)*strideB + j + N / 2] - T1[idx];
			//T3 = B22 - B12
			T3[idx] = matB[(i + K / 2)*strideB + j + N / 2] - matB[i*strideB + j + N / 2];
			//T4 = T2 - B21
			T4[idx] = T2[idx] - matB[(i + K / 2)*strideB + j];
		}
	}

	//M1 = A11*B11
	std::vector<float> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(float));
		mm_winograd(matA, matB, &M1[0], M / 2, N / 2, K / 2,
			strideA, strideB, N / 2);
	}

	//M2 = A12*B21
	std::vector<float> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(float));
		mm_winograd(matA + K / 2, matB + K*strideB/2, &M2[0], M / 2, N / 2, K / 2,
			strideA, strideB, N / 2);
	}

	//M3 = S4*B22
	std::vector<float> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(float));
		mm_winograd(&S4[0], matB + K*strideB/2 + N / 2, &M3[0], M / 2, N / 2, K / 2,
			K/2, strideB, N / 2);
	}

	//M4 = A22*T4
	std::vector<float> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(float));
		mm_winograd(matA + M*strideA / 2 + K / 2, &T4[0], &M4[0], M / 2, N / 2, K / 2,
			strideA, N / 2, N / 2);
	}

	//M5 = S1*T1
	std::vector<float> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(float));		
		mm_winograd(&S1[0], &T1[0], &M5[0], M / 2, N / 2, K / 2,
			K / 2, N/2, N / 2);
	}

	//M6 = S2*T2
	std::vector<float> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(float));
		mm_winograd(&S2[0], &T2[0], &M6[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}

	//M7 = S3*T3
	std::vector<float> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(float));		
		mm_winograd(&S3[0], &T3[0], &M7[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}

	for (int i = 0; i < M / 2; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			const int idx = i*N / 2 + j;
			//U1 = M1 + M2
			const auto U1 = M1[idx] + M2[idx];
			//U2 = M1 + M6
			const auto U2 = M1[idx] + M6[idx];
			//U3 = U2 + M7
			const auto U3 = U2 + M7[idx];
			//U4 = U2 + M5
			const auto U4 = U2 + M5[idx];
			//U5 = U4 + M3
			const auto U5 = U4 + M3[idx];
			//U6 = U3 - M4
			const auto U6 = U3 - M4[idx];
			//U7 = U3 + M5
			const auto U7 = U3 + M5[idx];

			//C11 = U1
			matC[i*strideC + j] = U1;
			//C12 = U5
			matC[i*strideC + j + N / 2] = U5;
			//C21 = U6
			matC[(i + M / 2)*strideC + j] = U6;
			//C22 = U7
			matC[(i + M / 2)*strideC + j + N / 2] = U7;
		}
	}
}

int main(int argc, char **argv) {
  // Do NOT count the time of reading files, malloc, and memset
  F0 = atoi(argv[1]);
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);
  readFloat(argv[5], X0, v_num * F0);
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();

  // Preprocessing time should be included

  somePreprocessing();

  edgeNormalization();

  // printf("Layer1 XW\n");
  XW(F0, F1, X0, X1_inter, W1);

  //mm_winograd(X0, W1, X1_inter, v_num, F1, F0, F0, F1, F1);

  // printf("Layer1 AX\n");
  AX(F1, X1_inter, X1);

  // unknown

  // printf("Layer1 ReLU\n");
  ReLU(F1, X1);

  // printf("Layer2 XW\n");
  XW(F1, F2, X1, X2_inter, W2);

  //mm_winograd(X1, W2, X2_inter, v_num, F2, F1, F1, F2, F2);

  // printf("Layer2 AX\n");
  AX(F2, X2_inter, X2);

  // unknown

  // printf("Layer2 LogSoftmax\n");
  LogSoftmax(F2, X2);

  // You need to compute the max row sum for result verification
  float max_sum = MaxRowSum(X2, F2);

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;

  // Finally, the max row sum and the computing time
  // should be print to the terminal in the following format
  printf("%.8f\n", max_sum);
  printf("%.8lf\n", l_timeMs);

  // Remember to free your allocated memory
  freeFloats();
}