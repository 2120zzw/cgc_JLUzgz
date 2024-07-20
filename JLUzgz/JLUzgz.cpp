#include <math.h>
#include <omp.h>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


// 正确结果：-16.68968964
using namespace std;

#define USEOPENMP


typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

vector<int> row_ptr;     // 行指针数组
vector<int> col_indices; // 列索引数组
vector<float> values;    // 值数组


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

void edgeNormalization() {
    #pragma omp parallel for
    for (int i = 0; i < v_num; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int dst = col_indices[j];
            float val = 1.0 / sqrt(degree[i]) / sqrt(degree[dst]);
            values[j] = val;
        }
    }
}

void raw_graph_to_CSR() {
    int edge_count = raw_graph.size() / 2;
    values.resize(edge_count, 0); // 初始分配空间，后续在归一化时更新
    col_indices.resize(edge_count);
    row_ptr.resize(v_num + 1, 0);
    degree.resize(v_num, 0);

    std::vector<int> src(edge_count);
    std::vector<int> dst(edge_count);

    // Step 1: 统计每个顶点的出度
    #pragma omp parallel for
    for (int i = 0; i < edge_count; i++) {
        src[i] = raw_graph[2 * i];
        dst[i] = raw_graph[2 * i + 1];
        #pragma omp atomic
        degree[src[i]]++;
    }

    // Step 2: 计算row_ptr数组
    for (int i = 1; i <= v_num; i++) {
        row_ptr[i] = row_ptr[i - 1] + degree[i - 1];
    }

    // Step 3: 填充col_indices数组
    std::vector<int> current_position = row_ptr; // 复制row_ptr用于当前插入位置追踪
    #pragma omp parallel for
    for (int i = 0; i < edge_count; i++) {
        int index = __sync_fetch_and_add(&current_position[src[i]], 1);
        col_indices[index] = dst[i];
    }

    // Step 4: 边归一化
    edgeNormalization();
}


void transpose_matrix(const float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
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

// void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
//   float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
//   float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
//   float(*tmp_W)[out_dim] = (float(*)[out_dim])W;
//   #ifdef USEOPENMP
//   #pragma omp parallel for
//   #endif
//   for (int i = 0; i < v_num; i++) {
//     for (int j = 0; j < out_dim; j++) {
//       for (int k = 0; k < in_dim; k++) {
//         tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
//       }
//     }
//   }
// }


void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W_T) {
    // 将一维数组转换为二维数组
    float (*tmp_in_X)[in_dim] = (float (*)[in_dim]) in_X;
    float (*tmp_out_X)[out_dim] = (float (*)[out_dim]) out_X;
    float (*tmp_W_T)[in_dim] = (float (*)[in_dim]) W_T; // 注意这里是转置后的矩阵

    // 初始化out_X矩阵为零
    std::memset(out_X, 0, v_num * out_dim * sizeof(float));

    // 使用OpenMP并行化外层循环
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < v_num; i++) {
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f; // 使用float类型的sum进行累加

            // 使用AVX-512指令进行向量化
            int k = 0;
#ifdef __AVX512F__
            __m512 sum512 = _mm512_setzero_ps();
            for (; k <= in_dim - 16; k += 16) {
                __m512 in_x512 = _mm512_loadu_ps(&tmp_in_X[i][k]);
                __m512 w512 = _mm512_loadu_ps(&tmp_W_T[j][k]);
                sum512 = _mm512_fmadd_ps(in_x512, w512, sum512);
            }
            // 累加AVX-512寄存器中的值
            float temp[16];
            _mm512_storeu_ps(temp, sum512);
            for (int t = 0; t < 16; t++) {
                sum += temp[t];
            }
#endif
            // 处理剩余的元素
            for (; k < in_dim; k++) {
                sum += tmp_in_X[i][k] * tmp_W_T[j][k];
            }

            tmp_out_X[i][j] = sum; // 将结果存储为float类型
        }
    }
}



// void AX(int dim, float *in_X, float *out_X) {
//   float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
//   float(*tmp_out_X)[dim] = (float(*)[dim])out_X;
//   for (int i = 0; i < v_num; i++) {
//     vector<int> &nlist = edge_index[i];
//     for (int j = 0; j < nlist.size(); j++) {
//       int nbr = nlist[j];
//       for (int k = 0; k < dim; k++) {
//         tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
//       }
//     }
//   }
// }


// void AX_CSR(int dim, float *in_X, float *out_X) {
//     float (*tmp_in_X)[dim] = (float (*)[dim]) in_X;
//     float (*tmp_out_X)[dim] = (float (*)[dim]) out_X;

//     // 初始化out_X矩阵为零
//     std::memset(out_X, 0, v_num * dim * sizeof(float));

//     #pragma omp parallel for
//     for (int i = 0; i < v_num; i++) {
//         for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
//             int nbr = col_indices[j];
//             float val = values[j];

//             // 手动向量化和循环展开
//             int k;
//             for (k = 0; k <= dim - 4; k += 4) {
//                 tmp_out_X[i][k] += tmp_in_X[nbr][k] * val;
//                 tmp_out_X[i][k + 1] += tmp_in_X[nbr][k + 1] * val;
//                 tmp_out_X[i][k + 2] += tmp_in_X[nbr][k + 2] * val;
//                 tmp_out_X[i][k + 3] += tmp_in_X[nbr][k + 3] * val;
//             }
//             // 处理剩余的元素
//             for (; k < dim; k++) {
//                 tmp_out_X[i][k] += tmp_in_X[nbr][k] * val;
//             }
//         }
//     }
// }

void AX_CSR(int dim, float *in_X, float *out_X) {
    float (*tmp_in_X)[dim] = (float (*)[dim]) in_X;
    float (*tmp_out_X)[dim] = (float (*)[dim]) out_X;

    // 初始化out_X矩阵为零
    std::memset(out_X, 0, v_num * dim * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < v_num; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int nbr = col_indices[j];
            float val = values[j];

            int k = 0;
#ifdef __AVX512F__
            // 使用AVX-512指令进行向量化
            __m512 val512 = _mm512_set1_ps(val);
            for (; k <= dim - 16; k += 16) {
                __m512 in_x512 = _mm512_loadu_ps(&tmp_in_X[nbr][k]);
                __m512 out_x512 = _mm512_loadu_ps(&tmp_out_X[i][k]);
                out_x512 = _mm512_fmadd_ps(in_x512, val512, out_x512);
                _mm512_storeu_ps(&tmp_out_X[i][k], out_x512);
            }
#endif
#ifdef __AVX2__
            // 使用AVX2指令进行向量化
            __m256 val256 = _mm256_set1_ps(val);
            for (; k <= dim - 8; k += 8) {
                __m256 in_x256 = _mm256_loadu_ps(&tmp_in_X[nbr][k]);
                __m256 out_x256 = _mm256_loadu_ps(&tmp_out_X[i][k]);
                out_x256 = _mm256_fmadd_ps(in_x256, val256, out_x256);
                _mm256_storeu_ps(&tmp_out_X[i][k], out_x256);
            }
#endif
#ifdef __AVX__
            // 使用AVX指令进行向量化
            __m128 val128 = _mm_set1_ps(val);
            for (; k <= dim - 4; k += 4) {
                __m128 in_x128 = _mm_loadu_ps(&tmp_in_X[nbr][k]);
                __m128 out_x128 = _mm_loadu_ps(&tmp_out_X[i][k]);
                out_x128 = _mm_fmadd_ps(in_x128, val128, out_x128);
                _mm_storeu_ps(&tmp_out_X[i][k], out_x128);
            }
#endif
            // 处理剩余的元素
            for (; k < dim; k++) {
                tmp_out_X[i][k] += tmp_in_X[nbr][k] * val;
            }
        }
    }
}


// void ReLU(int dim, float *X) {
//   for (int i = 0; i < v_num * dim; i++)
//     if (X[i] < 0) X[i] = 0;
// }


void ReLU(int dim, float *X) {
    int total_size = v_num * dim;

    #pragma omp parallel for
    for (int i = 0; i < total_size; i += 16) {
        // 使用AVX-512指令进行向量化处理
#ifdef __AVX512F__
        __m512 x = _mm512_loadu_ps(&X[i]);
        __m512 zeros = _mm512_setzero_ps();
        __m512 result = _mm512_max_ps(x, zeros);
        _mm512_storeu_ps(&X[i], result);
#endif
    }

    // 处理剩余的元素
    #pragma omp parallel for
    for (int i = (total_size / 16) * 16; i < total_size; i++) {
        if (X[i] < 0) X[i] = 0;
    }
}


void LogSoftmax(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

  #pragma omp parallel for
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
  
  #pragma omp parallel for
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
  raw_graph_to_CSR();
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

  // printf("Layer1 XW\n");
  float *W1_T = (float *)malloc(F0 * F1 * sizeof(float));
  transpose_matrix(W1, W1_T, F0, F1);
  XW(F0, F1, X0, X1_inter, W1_T);

  // printf("Layer1 AX\n");
  // AX(F1, X1_inter, X1);
  AX_CSR(F1, X1_inter, X1);

  // printf("Layer1 ReLU\n");
  ReLU(F1, X1);

  // printf("Layer2 XW\n");
  float *W2_T = (float *)malloc(F1 * F2 * sizeof(float));
  transpose_matrix(W2, W2_T, F1, F2);
  XW(F1, F2, X1, X2_inter, W2_T);

  // printf("Layer2 AX\n");
  // AX(F2, X2_inter, X2);
  AX_CSR(F2, X2_inter, X2);

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