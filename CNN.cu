#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "CNN.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMBER_THREAD 32

__global__ void ::Activate(int option, int number_neuron, float neuron[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_neuron){
		if(option == 0){
			neuron[j] = 2 / (1 + exp(-2 * neuron[j])) - 1;
		}
		else
		if(option == 1){
			neuron[j] = 1 / (1 + exp(-neuron[j]));
		}
		else
		if(option == 2){
			neuron[j] *= (neuron[j] > 0);
		}
	}
}
__global__ void ::Add(int number_memory, float A[], float B[], float C[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		C[j] = A[j] + B[j];
	}
}
__global__ void ::Adjust_Parameter(int batch_size, int lower_layer_index, int layer_index, int length_filter, int stride, int length_map[], int number_map[], float derivative[], float lower_neuron[], float weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int lower_i	= lower_layer_index;

	int j = index / ((number_map[lower_i] + 1) * length_filter * length_filter);

	if(j < number_map[i]){
		int m = ((index % ((number_map[lower_i] + 1) * length_filter * length_filter)) / (length_filter * length_filter));
		int n = ((index % ((number_map[lower_i] + 1) * length_filter * length_filter)) % (length_filter * length_filter)) / length_filter;
		int o = ((index % ((number_map[lower_i] + 1) * length_filter * length_filter)) % (length_filter * length_filter)) % length_filter;

		if(m < number_map[lower_i]){
			float sum = 0;

			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						int index[2] = {k * stride + n, l * stride + o};

						if(index[0] < length_map[lower_i] && index[1] < length_map[lower_i]){
							sum += derivative[h * number_map[i] * length_map[i] * length_map[i] +
											  j * length_map[i] * length_map[i] +
											  k * length_map[i] +
											  l]
								* lower_neuron[h * number_map[lower_i] * length_map[lower_i] * length_map[lower_i] +
											   m * length_map[lower_i] * length_map[lower_i] +
											   index[0] * length_map[lower_i] +
											   index[1]];
						}
					}
				}
			}
			weight[index] -= sum;
		}
		else
		if(m == number_map[lower_i] && n == 0 && o == 0){
			float sum = 0;

			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						sum += derivative[h * number_map[i] * length_map[i] * length_map[i] +
										  j * length_map[i] * length_map[i] +
										  k * length_map[i] +
										  l];
					}
				}
			}
			weight[index] -= sum;
		}
	}
}
__global__ void ::Backpropagate(int option, int batch_size, int layer_index, int upper_layer_index, int length_upper_filter, int upper_stride, int length_map[], int number_map[], float derivative[], float upper_derivative[], float upper_weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int upper_i	= upper_layer_index;

	int h = index / (number_map[i] * length_map[i] * length_map[i]);

	if(h < batch_size){
		int j = ((index % (number_map[i] * length_map[i] * length_map[i])) / (length_map[i] * length_map[i]));
		int k = ((index % (number_map[i] * length_map[i] * length_map[i])) % (length_map[i] * length_map[i])) / length_map[i];
		int l = ((index % (number_map[i] * length_map[i] * length_map[i])) % (length_map[i] * length_map[i])) % length_map[i];

		if(option == 0){
			int ks				= k / upper_stride;
			int ls				= l / upper_stride;
			int neuron_index[2] = {ks - (length_upper_filter - 1), ls - (length_upper_filter - 1)};

			float sum = 0;

			if(neuron_index[0] < 0) neuron_index[0] = 0;
			if(neuron_index[1] < 0) neuron_index[1] = 0;

			for(int m = 0;m < number_map[upper_i];m++){
				for(int n = neuron_index[0];n < length_map[upper_i] && n <= ks;n++){
					for(int o = neuron_index[1];o < length_map[upper_i] && o <= ls;o++){
						sum += upper_derivative[h * number_map[upper_i] * length_map[upper_i] * length_map[upper_i] +
												m * length_map[upper_i] * length_map[upper_i] +
												n * length_map[upper_i] +
												o]
							* upper_weight[m * (number_map[i] + 1) * length_upper_filter * length_upper_filter +
										   j * length_upper_filter * length_upper_filter +
										   abs(ks - n) * length_upper_filter +
										   abs(ls - o)];
					}
				}
			}
			derivative[index] = sum;
		}
		else
		if(option == 1){
			int stride = length_map[i] / length_map[i + 1];

			derivative[index] = upper_derivative[h * number_map[i + 1] * length_map[i + 1] * length_map[i + 1] +
												 j * length_map[i + 1] * length_map[i + 1] +
												 (k / stride) * length_map[i + 1] +
												 (l / stride)];
		}
		else
		if(option == 2){
			int margin = (length_map[i + 1] - length_map[i]) / 2;

			derivative[index] = upper_derivative[h * number_map[i + 1] * length_map[i + 1] * length_map[i + 1] +
												 j * length_map[i + 1] * length_map[i + 1] +
												 (margin + k) * length_map[i + 1] +
												 (margin + l)];
		}
	}
}
__global__ void ::Batch_Normalization_Activate(int option, int batch_size, int length_map, int number_map, float epsilon, float gamma[], float beta[], float mean[], float variance[], float sum_mean[], float sum_variance[], float neuron[], float neuron_batch_0[], float neuron_batch_1[]){
	int j = blockIdx.x;

	if(option == 0){
		__shared__ float sum[NUMBER_THREAD];

		sum[threadIdx.x] = 0;
		for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
			int h = m / (length_map * length_map);
			int k = m % (length_map * length_map);

			sum[threadIdx.x] += neuron[h * number_map * length_map * length_map +
									   j * length_map * length_map +
									   k];
		}
		for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
			__syncthreads();

			if(threadIdx.x < m){
				sum[threadIdx.x] += sum[threadIdx.x + m];
			}
		}
		if(threadIdx.x == 0){
			sum_mean[j] += (mean[j] = sum[0] / (batch_size * length_map * length_map));
		}
		__syncthreads();

		sum[threadIdx.x] = 0;
		for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
			int h = m / (length_map * length_map);
			int k = m % (length_map * length_map);

			int index = h * number_map * length_map * length_map +
						j * length_map * length_map +
						k;

			sum[threadIdx.x] += (neuron[index] - mean[j]) * (neuron[index] - mean[j]);
		}
		for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
			__syncthreads();

			if(threadIdx.x < m){
				sum[threadIdx.x] += sum[threadIdx.x + m];
			}
		}
		if(threadIdx.x == 0){
			sum_variance[j] += (variance[j] = sum[0] / (batch_size * length_map * length_map));
		}
		__syncthreads();

		for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
			int h = m / (length_map * length_map);
			int k = m % (length_map * length_map);

			int index = h * number_map * length_map * length_map +
						j * length_map * length_map +
						k;

			neuron_batch_0[index]	= (neuron[index] - mean[j]) / sqrt(variance[j] + epsilon);
			neuron_batch_1[index]	= neuron[index];
			neuron[index]			= gamma[j] * neuron_batch_0[index] + beta[j];
		}
	}
	else
	if(option == 1){
		for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
			int h = m / (length_map * length_map);
			int k = m % (length_map * length_map);

			int index = h * number_map * length_map * length_map +
						j * length_map * length_map +
						k;

			float stdv = sqrt(variance[j] + epsilon);

			neuron[index] = gamma[j] / stdv * neuron[index] + (beta[j] - gamma[j] * mean[j] / stdv);
		}
	}
}
__global__ void ::Batch_Normalization_Adjust_Parameter(int batch_size, int length_map, int number_map, float gamma[], float beta[], float derivative_batch_1[], float neuron_batch_0[]){
	int j = blockIdx.x;

	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
		int h = m / (length_map * length_map);
		int k = m % (length_map * length_map);

		int index = h * number_map * length_map * length_map +
					j * length_map * length_map +
					k;

		sum[threadIdx.x] += derivative_batch_1[index] * neuron_batch_0[index];
	}
	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		gamma[j] -= sum[0] / (length_map * length_map);
	}

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
		int h = m / (length_map * length_map);
		int k = m % (length_map * length_map);

		sum[threadIdx.x] += derivative_batch_1[h * number_map * length_map * length_map +
											   j * length_map * length_map +
											   k];
	}
	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		beta[j] -= sum[0] / (length_map * length_map);
	}
}
__global__ void ::Batch_Normalization_Differentiate(int batch_size, int length_map, int number_map, float epsilon, float gamma[], float beta[], float mean[], float variance[], float derivative[], float derivative_batch_0[], float derivative_batch_1[], float neuron_batch_1[]){
	int j = blockIdx.x;

	__shared__ float derivative_mean;
	__shared__ float derivative_variance;
	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
		int h = m / (length_map * length_map);
		int k = m % (length_map * length_map);

		int index = h * number_map * length_map * length_map +
					j * length_map * length_map +
					k;

		derivative_batch_0[index] = derivative[index] * gamma[j];
		sum[threadIdx.x] += derivative_batch_0[index] * (neuron_batch_1[index] - mean[j]);
	}
	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		derivative_variance = sum[0] * (-0.5) * pow(variance[j] + epsilon, (float)(-1.5));
	}

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
		int h = m / (length_map * length_map);
		int k = m % (length_map * length_map);

		sum[threadIdx.x] += derivative_batch_0[h * number_map * length_map * length_map + j * length_map * length_map + k];
	}
	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		derivative_mean = -sum[0] / sqrt(variance[j] + epsilon);
	}
	__syncthreads();

	for(int m = threadIdx.x;m < batch_size * length_map * length_map;m += blockDim.x){
		int h = m / (length_map * length_map);
		int k = m % (length_map * length_map);

		int index = h * number_map * length_map * length_map + j * length_map * length_map + k;

		derivative_batch_1[index]	= derivative[index];
		derivative[index]			= derivative_batch_0[index] / sqrt(variance[j] + epsilon) + derivative_variance * 2 * (neuron_batch_1[index] - mean[j]) / (batch_size * length_map * length_map) + derivative_mean / (batch_size * length_map * length_map);
	}
}
__global__ void ::Calculate_Loss(int option, int number_neuron, float *loss, float neuron[], float target_output[]){
	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int j = threadIdx.x;j < number_neuron;j += blockDim.x){
		if(option == 0){
			sum[threadIdx.x] -= target_output[j] * log(neuron[j] + 0.000001) + (1 - target_output[j]) * log(1 - neuron[j] + 0.000001);
		}
		else
		if(option == 1){
			sum[threadIdx.x] += 0.5 * (neuron[j] - target_output[j]) * (neuron[j] - target_output[j]);
		}
	}

	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		(*loss) += sum[0];
	}
}
__global__ void ::Differentiate(int option, int number_memory, float learning_rate, float derivative[], float neuron[], float target_output[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		if(option == 0){
			derivative[j] *= (1 - neuron[j]) * (1 + neuron[j]);
		}
		else
		if(option == 1){
			derivative[j] *= (1 - neuron[j]) * neuron[j];
		}
		else
		if(option == 2){
			derivative[j] *= (neuron[j] > 0);
		}
		else
		if(option == 3){
			derivative[j] = learning_rate * (neuron[j] - target_output[j]);
		}
	}
}
__global__ void ::Dropout(int option, int batch_size, int number_map, int length_map, int seed, float rate, float neuron[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int h = (index / (number_map * length_map * length_map));
	int j = (index % (number_map * length_map * length_map)) / (length_map * length_map);

	if(j < number_map){
		curandState s[NUMBER_THREAD];

		if(option == 0){
			curand_init(seed + h * number_map * length_map * length_map + j, 0, 0, &s[threadIdx.x]);

			neuron[index] *= (curand_uniform(&s[threadIdx.x]) <= rate);
		}
		else
		if(option == 1){
			neuron[index] *= rate;
		}
	}
}
__global__ void ::Feedforward(int option, int batch_size, int lower_layer_index, int layer_index, int length_filter, int stride, int number_map[], int length_map[], float lower_neuron[], float neuron[], float weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int lower_i	= lower_layer_index;

	int h = index / (number_map[i] * length_map[i] * length_map[i]);

	if(h < batch_size){
		int j = ((index % (number_map[i] * length_map[i] * length_map[i])) / (length_map[i] * length_map[i]));
		int k = ((index % (number_map[i] * length_map[i] * length_map[i])) % (length_map[i] * length_map[i])) / length_map[i];
		int l = ((index % (number_map[i] * length_map[i] * length_map[i])) % (length_map[i] * length_map[i])) % length_map[i];

		if(option == 0){
			float sum = 0;

			for(int m = 0;m < number_map[lower_i];m++){
				for(int n = 0;n < length_filter;n++){
					for(int o = 0;o < length_filter;o++){
						int neuron_index[2] = {k * stride + n, l * stride + o};

						if(neuron_index[0] < length_map[lower_i] && neuron_index[1] < length_map[lower_i]){
							sum += lower_neuron[h * number_map[lower_i] * length_map[lower_i] * length_map[lower_i] +
												m * length_map[lower_i] * length_map[lower_i] +
												neuron_index[0] * length_map[lower_i] +
												neuron_index[1]]
								* weight[j * (number_map[lower_i] + 1) * length_filter * length_filter +
										 m * length_filter * length_filter +
										 n * length_filter +
										 o];
						}
					}
				}
			}
			neuron[index] = sum + weight[j * (number_map[lower_i] + 1) * length_filter * length_filter +
										 number_map[lower_i] * length_filter * length_filter];
		}
		else
		if(option == 1){
			int stride = length_map[i - 1] / length_map[i];

			float sum = 0;
						
			for(int m = 0;m < stride;m++){
				for(int n = 0;n < stride;n++){
					sum += lower_neuron[h * number_map[i - 1] * length_map[i - 1] * length_map[i - 1] +
										j * length_map[i - 1] * length_map[i - 1] +
										(k * stride + m) * length_map[i - 1] +
										(l * stride + n)];
				}
			}
			neuron[index] = sum / (stride * stride);
		}
		else
		if(option == 2){
			int stride = length_map[i - 1] / length_map[i];

			float max = -1;

			for(int m = 0;m < stride;m++){
				for(int n = 0;n < stride;n++){
					int neuron_index = h * number_map[i - 1] * length_map[i - 1] * length_map[i - 1] +
									   j * length_map[i - 1] * length_map[i - 1] +
									   (k * stride + m) * length_map[i - 1] +
									   (l * stride + n);

					if(max < lower_neuron[neuron_index]){
						max = lower_neuron[neuron_index];
					}
				}
			}
			neuron[index] = max;
		}
		else
		if(option == 3){
			int margin = (length_map[i] - length_map[i - 1]) / 2;

			if(k < length_map[i - 1] && l < length_map[i - 1]){
				neuron[h * number_map[i] * length_map[i] * length_map[i] +
					   j * length_map[i] * length_map[i] +
					   (margin + k) * length_map[i] +
					   (margin + l)]
				= lower_neuron[h * number_map[i - 1] * length_map[i - 1] * length_map[i - 1] +
							   j * length_map[i - 1] * length_map[i - 1] +
							   k * length_map[i - 1] +
							   l];
			}
			else{
				neuron[index] = 0;
			}
		}
	}
}
__global__ void ::Multiply(int number_memory, float A[], float value, float B[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		B[j] = A[j] * value;
	}
}
__global__ void ::Multiply(int number_memory, float A[], float B[], float C[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		C[j] = A[j] * B[j];
	}
}
__global__ void ::Randomize(int number_memory, int seed, float scale, float shift, float A[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		curandState s[NUMBER_THREAD];

		curand_init(seed + j, 0, 0, &s[threadIdx.x]);
		A[j] = scale * curand_uniform(&s[threadIdx.x]) + shift;
	}
}
__global__ void ::Set(int number_memory, float value, float A[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_memory){
		A[j] = value;
	}
}
__global__ void ::Softmax(int batch_size, int length_map, int number_map, float neuron[]){
	int h = blockIdx.x * blockDim.x + threadIdx.x;

	if(h < batch_size){
		float max = 0;
		float sum = 0;

		for(int j = 0;j < number_map;j++){
			int index = h * number_map * length_map * length_map +
						j * length_map * length_map;

			if(max < neuron[index]){
				max = neuron[index];
			}
		}
		for(int j = 0;j < number_map;j++){
			int index = h * number_map * length_map * length_map +
						j * length_map * length_map;

			sum += (neuron[index] = exp(neuron[index] - max));
		}
		for(int j = 0;j < number_map;j++){
			neuron[h * number_map * length_map * length_map +
				   j * length_map * length_map] /= sum;
		}
	}
}

void Convolutional_Neural_Networks_CUDA::Activate(char option[], int layer_index){
	int i = layer_index;

	int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

	float *neuron = this->neuron[0][i];

	if(strstr(type_layer[i], "bn")){
		Batch_Normalization_Activate(option, "normal", layer_index);
	}
	if(strstr(type_layer[i], "sc")){
		int lower_layer_index = i - atoi(strstr(type_layer[i], "sc") + 2);

		if(lower_layer_index < i){
			if(strstr(type_layer[i], "psc") && strstr(type_layer[i], "bn")){
				Batch_Normalization_Activate(option, "shortcut", layer_index);
			}
			::Add<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_memory, this->neuron[1][i], neuron, neuron);
		}
	}

	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "ht")){
			::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, number_memory, neuron);
		}
		else
		if(strstr(type_layer[i], "ls")){
			::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, neuron);
		}
		else{
			::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, number_memory, neuron);
		}

		if(strstr(type_layer[i], "do")){
			char *rate = strstr(type_layer[i], "do") + 2;

			if(!strcmp(option, "train")){
				Dropout<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, number_map[i], length_map[i], clock(), atof(rate), neuron);
			}
			else
			if(!strcmp(option, "test")){
				Dropout<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, number_map[i], length_map[i], clock(), atof(rate), neuron);
			}
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "ce")){
			if(strstr(type_layer[i], "sm")){
				::Softmax<<<batch_size / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, length_map[i], number_map[i], neuron);
			}
			else{
				::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, neuron);
			}
		}
		else
		if(strstr(type_layer[i], "mse")){
			if(strstr(type_layer[i], "ht")){
				::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, number_memory, neuron);
			}
			else
			if(strstr(type_layer[i], "ia")){
				// neuron = neuron
			}
			else{
				::Activate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, neuron);
			}
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Adjust_Parameter(int layer_index){
	int i = layer_index;

	int number_parameter = number_map[i] * (number_map[i - 1] + 1) * length_filter[i] * length_filter[i];

	float *derivative	= this->derivative[0][i];
	float *lower_neuron	= this->neuron[0][i - 1];
	float *weight		= this->weight[0][i];

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Adjust_Parameter("normal", layer_index);
		}
		::Adjust_Parameter<<<number_parameter / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, i - 1, layer_index, length_filter[i], stride[i], length_map_factor, number_map_factor, derivative, lower_neuron, weight);

		if(strstr(type_layer[i], "psc")){
			int lower_layer_index = i - atoi(strstr(type_layer[i], "psc") + 2);

			lower_neuron	= this->neuron[0][lower_layer_index];
			derivative		= this->derivative[1][i];
			weight			= this->weight[1][i];

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Adjust_Parameter("shortcut", layer_index);
			}
			::Adjust_Parameter<<<number_parameter / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, lower_layer_index, layer_index, shortcut_length_filter[i], shortcut_stride[i], length_map_factor, number_map_factor, derivative, lower_neuron, weight);
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Backpropagate(int layer_index){
	if(layer_index == number_layer - 1){
		return;
	}

	int i = layer_index;

	int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

	float *derivative		= this->derivative[0][i];
	float *upper_derivative	= this->derivative[0][i + 1];
	float *upper_weight		= this->weight[0][i + 1];

	if(type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L'){
		::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, layer_index, i + 1, length_filter[i + 1], stride[i + 1], length_map_factor, number_map_factor, derivative, upper_derivative, upper_weight);
	}
	else
	if(type_layer[i + 1][0] == 'P'){
		if(strstr(type_layer[i + 1], "avg") || strstr(type_layer[i + 1], "max")){
			::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, layer_index, i + 1, length_filter[i + 1], stride[i + 1], length_map_factor, number_map_factor, derivative, upper_derivative, upper_weight);
		}
		else
		if(strstr(type_layer[i + 1], "pad")){
			::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, batch_size, layer_index, i + 1, length_filter[i + 1], stride[i + 1], length_map_factor, number_map_factor, derivative, upper_derivative, upper_weight);
		}
	}

	if(strstr(type_layer[i], "sc")){
		int upper_layer_index = i;

		for(int upper_i = i + 1;upper_i < number_layer;upper_i++){
			char *type = strstr(type_layer[upper_i], "sc");

			if(type && upper_i == i + atoi(strstr(type, "sc") + 2)){
				upper_layer_index = upper_i;
				break;
			}
		}
		if(upper_layer_index > i){
			derivative		 = this->derivative[1][i];
			upper_derivative = this->derivative[1][upper_layer_index];
			upper_weight	 = this->weight[1][upper_layer_index];

			if(strstr(type_layer[upper_layer_index], "psc")){
				::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, layer_index, upper_layer_index, shortcut_length_filter[upper_layer_index], shortcut_stride[upper_layer_index], length_map_factor, number_map_factor, derivative, upper_derivative, upper_weight);
			}
			else{
				cudaMemcpy(derivative, upper_derivative, sizeof(float) * number_memory, cudaMemcpyDeviceToDevice);
			}
			Add<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_memory, derivative, this->derivative[0][i], this->derivative[0][i]);
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Differentiate(int layer_index, float learning_rate, float target_output[]){
	int i = layer_index;

	int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

	float *derivative	= this->derivative[0][i];
	float *neuron		= this->neuron[0][i];


	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "ht")){
			::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, number_memory, learning_rate, derivative, neuron, target_output);
		}
		else
		if(strstr(type_layer[i], "ls")){
			::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, learning_rate, derivative, neuron, target_output);
		}
		else{
			::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, number_memory, learning_rate, derivative, neuron, target_output);
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(3, number_memory, learning_rate, derivative, neuron, target_output);

		if(strstr(type_layer[i], "ce")){
			if(strstr(type_layer[i], "sm")){
				// derivative = derivative;
			}
			else{
				::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, learning_rate, derivative, neuron, target_output);
			}
		}
		else
		if(strstr(type_layer[i], "mse")){
			if(strstr(type_layer[i], "ht")){
				::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, number_memory, learning_rate, derivative, neuron, target_output);
			}
			else
			if(strstr(type_layer[i], "ia")){
				// derivative *= 1;
			}
			else{
				::Differentiate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, number_memory, learning_rate, derivative, neuron, target_output);
			}
		}
	}

	if(strstr(type_layer[i], "sc")){
		int lower_layer_index = i - atoi(strstr(type_layer[i], "sc") + 2);

		if(lower_layer_index < i){
			cudaMemcpy(this->derivative[1][i], derivative, sizeof(float) * number_memory, cudaMemcpyDeviceToDevice);

			if(strstr(type_layer[i], "psc") && strstr(type_layer[i], "bn")){
				Batch_Normalization_Differentiate("shortcut", layer_index);
			}
		}
	}
	if(strstr(type_layer[i], "bn")){
		Batch_Normalization_Differentiate("normal", layer_index);
	}
}
void Convolutional_Neural_Networks_CUDA::Feedforward(int layer_index){
	int i = layer_index;

	int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

	float *lower_neuron = this->neuron[0][i - 1];
	float *neuron		= this->neuron[0][i];
	float *weight		= this->weight[0][i];

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, i - 1, layer_index, length_filter[i], stride[i], number_map_factor, length_map_factor, lower_neuron, neuron, weight);
	}
	else
	if(type_layer[i][0] == 'P'){
		if(strstr(type_layer[i], "avg")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, i - 1, layer_index, length_filter[i], stride[i], number_map_factor, length_map_factor, lower_neuron, neuron, weight);
		}
		else
		if(strstr(type_layer[i], "max")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, batch_size, i - 1, layer_index, length_filter[i], stride[i], number_map_factor, length_map_factor, lower_neuron, neuron, weight);
		}
		else
		if(strstr(type_layer[i], "pad")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(3, batch_size, i - 1, layer_index, length_filter[i], stride[i], number_map_factor, length_map_factor, lower_neuron, neuron, weight);
		}
	}

	if(strstr(type_layer[i], "sc")){
		int lower_layer_index = i - atoi(strstr(type_layer[i], "sc") + 2);

		if(lower_layer_index < i){
			lower_neuron = this->neuron[0][lower_layer_index];
			neuron		 = this->neuron[1][i];
			weight		 = this->weight[1][i];

			if(strstr(type_layer[i], "psc")){
				::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, lower_layer_index, layer_index, shortcut_length_filter[i], shortcut_stride[i], number_map_factor, length_map_factor, lower_neuron, neuron, weight);
			}
			else{
				cudaMemcpy(neuron, lower_neuron, sizeof(float) * number_memory, cudaMemcpyDeviceToDevice);
			}
		}
	}
}

void Convolutional_Neural_Networks_CUDA::Batch_Normalization_Activate(char option[], char type[], int layer_index){
	int i = layer_index;

	float *gamma;
	float *beta;
	float *mean;
	float *variance;
	float *sum_mean;
	float *sum_variance;

	float *neuron;
	float *neuron_batch[2];

	if(!strcmp(type, "normal")){
		gamma			= this->gamma[0][i];
		beta			= this->beta[0][i];
		mean			= this->mean[0][i];
		variance		= this->variance[0][i];
		sum_mean		= this->sum_mean[0][i];
		sum_variance	= this->sum_variance[0][i];

		neuron			= this->neuron[0][i];
		neuron_batch[0]	= this->neuron[2][i];
		neuron_batch[1] = this->neuron[3][i];
	}
	else
	if(!strcmp(type, "shortcut")){
		gamma			= this->gamma[1][i];
		beta			= this->beta[1][i];
		mean			= this->mean[1][i];
		variance		= this->variance[1][i];
		sum_mean		= this->sum_mean[1][i];
		sum_variance	= this->sum_variance[1][i];

		neuron			= this->neuron[1][i];
		neuron_batch[0]	= this->neuron[4][i];
		neuron_batch[1] = this->neuron[5][i];
	}

	if(!strcmp(option, "train")){
		::Batch_Normalization_Activate<<<number_map[i], NUMBER_THREAD>>>(0, batch_size, length_map[i], number_map[i], epsilon, gamma, beta, mean, variance, sum_mean, sum_variance, neuron, neuron_batch[0], neuron_batch[1]);
	}
	else
	if(!strcmp(option, "test")){
		::Batch_Normalization_Activate<<<number_map[i], NUMBER_THREAD>>>(1, batch_size, length_map[i], number_map[i], epsilon, gamma, beta, mean, variance, sum_mean, sum_variance, neuron, neuron_batch[0], neuron_batch[1]);
	}
}
void Convolutional_Neural_Networks_CUDA::Batch_Normalization_Adjust_Parameter(char type[], int layer_index){
	int i = layer_index;

	float *gamma;
	float *beta;

	float *derivative_batch;
	float *neuron_batch;

	if(!strcmp(type, "normal")){
		gamma	= this->gamma[0][i];
		beta	= this->beta[0][i];

		derivative_batch	= this->derivative[3][i];
		neuron_batch		= this->neuron[2][i];
	}
	else
	if(!strcmp(type, "shortcut")){
		gamma	= this->gamma[1][i];
		beta	= this->beta[1][i];

		derivative_batch	= this->derivative[5][i];
		neuron_batch		= this->neuron[4][i];
	}

	::Batch_Normalization_Adjust_Parameter<<<number_map[i], NUMBER_THREAD>>>(batch_size, length_map[i], number_map[i], gamma, beta, derivative_batch, neuron_batch);
}
void Convolutional_Neural_Networks_CUDA::Batch_Normalization_Differentiate(char type[], int layer_index){
	int i = layer_index;

	float *gamma;
	float *beta;
	float *mean;
	float *variance;

	float *derivative;
	float *derivative_batch[2];
	float *neuron_batch[2];

	if(!strcmp(type, "normal")){
		gamma	 = this->gamma[0][i];
		beta	 = this->beta[0][i];
		mean	 = this->mean[0][i];
		variance = this->variance[0][i];

		derivative			= this->derivative[0][i];
		derivative_batch[0]	= this->derivative[2][i];
		derivative_batch[1] = this->derivative[3][i];
		neuron_batch[0]		= this->neuron[2][i];
		neuron_batch[1]		= this->neuron[3][i];
	}
	else
	if(!strcmp(type, "shortcut")){
		gamma	 = this->gamma[1][i];
		beta	 = this->beta[1][i];
		mean	 = this->mean[1][i];
		variance = this->variance[1][i];

		derivative			= this->derivative[1][i];
		derivative_batch[0]	= this->derivative[4][i];
		derivative_batch[1] = this->derivative[5][i];
		neuron_batch[0]		= this->neuron[4][i];
		neuron_batch[1]		= this->neuron[5][i];
	}

	::Batch_Normalization_Differentiate<<<number_map[i], NUMBER_THREAD>>>(batch_size, length_map[i], number_map[i], epsilon, gamma, beta, mean, variance, derivative, derivative_batch[0], derivative_batch[1], neuron_batch[1]);
}

void Convolutional_Neural_Networks_CUDA::Resize_Memory(int batch_size){
	if(this->batch_size != batch_size){
		for(int g = 0;g < number_memory_type;g++){
			for(int i = 0;i < number_layer;i++){
				if(Access_Memory(g, i)){
					int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

					cudaFree(derivative[g][i]);
					cudaFree(neuron[g][i]);

					cudaMalloc(&derivative[g][i],	sizeof(float) * number_memory);
					cudaMalloc(&neuron[g][i],		sizeof(float) * number_memory);

					if(number_memory / NUMBER_THREAD + 1 > 65535){
						fprintf(stderr, "[required gridDim: %d > 65535], (NUMBER_THREAD: %d) must be a higher value.\nplease refer to the CNN.cu/line 11\n", number_memory / NUMBER_THREAD + 1, NUMBER_THREAD);
					}
				}
			}
		}
		for(int h = 0;h < number_parameter_type;h++){
			for(int i = 0;i < number_layer;i++){
				if(Access_Parameter(h, i)){
					int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "psc") + 2)):(i - 1);
					int number_parameter	= number_map[i] * (number_map[lower_layer_index] + 1) * length_filter[i] * length_filter[i];

					if(number_parameter / NUMBER_THREAD + 1 > 65535){
						fprintf(stderr, "[required gridDim: %d > 65535], (NUMBER_THREAD: %d) must be a higher value.\nplease refer to the CNN.cu/line 11\n", number_parameter / NUMBER_THREAD + 1, NUMBER_THREAD);
					}
				}
			}
		}
		this->batch_size = batch_size;
	}
}

bool Convolutional_Neural_Networks_CUDA::Access_Memory(int type_index, int layer_index){
	int g = type_index;
	int i = layer_index;

	switch(g){
	case 0:
		return true;
	case 1:
		return (strstr(type_layer[i], "sc"));
	case 2:
	case 3:
		return (strstr(type_layer[i], "bn"));
	case 4:
	case 5:
		return (strstr(type_layer[i], "bn") && strstr(type_layer[i], "psc"));
	}
	return false;
}
bool Convolutional_Neural_Networks_CUDA::Access_Parameter(int type_index, int layer_index){
	int h = type_index;
	int i = layer_index;

	if(i > 0){
		switch(h){
		case 0:
			return (type_layer[i][0] == 'C') || (type_layer[i][0] == 'L');
		case 1:
			return (strstr(type_layer[i], "psc"));
		}
	}
	return false;
}

Convolutional_Neural_Networks_CUDA::Convolutional_Neural_Networks_CUDA(char **type_layer, int number_layer, int length_map[], int number_map[]){
	this->length_filter	= new int[number_layer];
	this->length_map	= new int[number_layer];
	this->number_layer	= number_layer;
	this->number_map	= new int[number_layer];
	this->stride		= new int[number_layer];
	this->type_layer	= new char*[number_layer];

	shortcut_length_filter	= new int[number_layer];
	shortcut_stride			= new int[number_layer];

	cudaMallocManaged(&length_map_factor, sizeof(int) * number_layer);
	cudaMallocManaged(&number_map_factor, sizeof(int) * number_layer);

	batch_size				= 1;
	number_memory_type		= 6;
	number_parameter_type	= 2;

	for(int i = 0;i < number_layer;i++){
		this->type_layer[i]	 = new char[strlen(type_layer[i]) + 1];
		strcpy(this->type_layer[i], type_layer[i]);
		this->number_map[i]	 = number_map[i];
		number_map_factor[i] = number_map[i];
		this->length_map[i]	 = (length_map == 0) ? (1):(length_map[i]);
		length_map_factor[i] = this->length_map[i];
	}
	for(int i = 1;i < number_layer;i++){
		char *type = strtok(this->type_layer[i], "/");

		if(strstr(type, "fs"))	length_filter[i] = atoi(strstr(type, "fs") + 2);
		else					length_filter[i] = length_map[i - 1] - length_map[i] + 1;

		if(strstr(type, "st"))	stride[i] = atoi(strstr(type, "st") + 2);
		else					stride[i] = 1;

		strcpy(this->type_layer[i], type_layer[i]);

		if(strstr(type_layer[i], "psc")){
			char *type_shortcut = strstr(type_layer[i], "psc");

			shortcut_length_filter[i] = 1;

			if(strstr(type_shortcut, "st"))	shortcut_stride[i] = atoi(strstr(type_shortcut, "st") + 2);
			else							shortcut_stride[i] = 1;
		}
	}

	gamma		 = new float**[number_parameter_type];
	beta		 = new float**[number_parameter_type];
	mean		 = new float**[number_parameter_type];
	variance	 = new float**[number_parameter_type];
	sum_mean	 = new float**[number_parameter_type];
	sum_variance = new float**[number_parameter_type];

	for(int h = 0;h < number_parameter_type;h++){
		gamma[h]		= new float*[number_layer];
		beta[h]			= new float*[number_layer];
		mean[h]			= new float*[number_layer];
		variance[h]		= new float*[number_layer];
		sum_mean[h]		= new float*[number_layer];
		sum_variance[h]	= new float*[number_layer];

		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				cudaMallocManaged(&gamma[h][i],			sizeof(float) * number_map[i]);
				cudaMallocManaged(&beta[h][i],			sizeof(float) * number_map[i]);
				cudaMallocManaged(&mean[h][i],			sizeof(float) * number_map[i]);
				cudaMallocManaged(&variance[h][i],		sizeof(float) * number_map[i]);
				cudaMallocManaged(&sum_mean[h][i],		sizeof(float) * number_map[i]);
				cudaMallocManaged(&sum_variance[h][i],	sizeof(float) * number_map[i]);
			}
		}
	}

	derivative	= new float**[number_memory_type];
	neuron		= new float**[number_memory_type];

	for(int g = 0;g < number_memory_type;g++){
		derivative[g]	= new float*[number_layer];
		neuron[g]		= new float*[number_layer];

		for(int i = 0;i < number_layer;i++){
			if(Access_Memory(g, i)){
				int number_memory = batch_size * number_map[i] * length_map[i] * length_map[i];

				cudaMalloc(&derivative[g][i],	sizeof(float) * number_memory);
				cudaMalloc(&neuron[g][i],		sizeof(float) * number_memory);
			}
		}
	}

	weight = new float**[number_parameter_type];

	for(int h = 0;h < number_parameter_type;h++){
		weight[h] = new float*[number_layer];

		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i)){
				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "psc") + 2)):(i - 1);
				int number_parameter	= number_map[i] * (number_map[lower_layer_index] + 1) * length_filter[i] * length_filter[i];

				cudaMallocManaged(&weight[h][i], sizeof(float) * number_parameter);
			}
		}
	}
}
Convolutional_Neural_Networks_CUDA::~Convolutional_Neural_Networks_CUDA(){
	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				cudaFree(gamma[h][i]);
				cudaFree(beta[h][i]);
				cudaFree(mean[h][i]);
				cudaFree(variance[h][i]);
				cudaFree(sum_mean[h][i]);
				cudaFree(sum_variance[h][i]);
			}
		}
		delete[] gamma[h];
		delete[] beta[h];
		delete[] mean[h];
		delete[] variance[h];
		delete[] sum_mean[h];
		delete[] sum_variance[h];
	}
	delete[] gamma;
	delete[] beta;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;

	for(int g = 0;g < number_memory_type;g++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Memory(g, i)){
				cudaFree(derivative[g][i]);
				cudaFree(neuron[g][i]);
			}
		}
		delete[] derivative[g];
		delete[] neuron[g];
	}
	delete[] derivative;
	delete[] neuron;

	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i)){
				cudaFree(weight[h][i]);
			}
		}
		delete[] weight[h];
	}
	delete[] weight;

	for(int i = 0;i < number_layer;i++){
		delete[] type_layer[i];
	}
	delete[] type_layer;

	delete[] length_filter;
	delete[] length_map;
	delete[] number_map;
	delete[] stride;

	delete[] shortcut_length_filter;
	delete[] shortcut_stride;

	cudaFree(length_map_factor);
	cudaFree(number_map_factor);
}

void Convolutional_Neural_Networks_CUDA::Initialize_Parameter(int seed){
	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i)){
				if(strstr(type_layer[i], "bn")){
					Set<<<number_map[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_map[i], 1, gamma[h][i]);
					Set<<<number_map[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_map[i], 0, beta[h][i]);
				}

				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
				int number_parameter	= number_map[i] * (number_map[lower_layer_index] + 1) * length_filter[i] * length_filter[i];

				Randomize<<<number_parameter / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_parameter, seed, 0.2, -0.1, weight[h][i]);
			}
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	fprintf(file, "%.12f\n", epsilon);

	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i)){
				if(strstr(type_layer[i], "bn")){
					for(int j = 0;j < number_map[i];j++){
						fprintf(file, "%f\n", gamma[h][i][j]);
						fprintf(file, "%f\n", beta[h][i][j]);
					}
				}

				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
				int number_parameter	= number_map[i] * (number_map[lower_layer_index] + 1) * length_filter[i] * length_filter[i];

				for(int j = 0;j < number_parameter;j++){
					fprintf(file, "%f\n", weight[h][i][j]);
				}
			}
		}
	}
	fclose(file);
}
void Convolutional_Neural_Networks_CUDA::Test(int batch_size, float **input, float **output){
	Resize_Memory(batch_size);

	for(int h = 0, i = 0;h < batch_size;h++){
		cudaMemcpy(&neuron[0][i][h * number_map[i] * length_map[i] * length_map[i]], input[h], sizeof(float) * number_map[i] * length_map[i] * length_map[i], cudaMemcpyHostToDevice);
	}

	for(int i = 1;i < number_layer;i++){
		Feedforward	(i);
		Activate	("test", i);
	}

	for(int h = 0, i = number_layer - 1;h < batch_size;h++){
		cudaMemcpy(output[h], &neuron[0][i][h * number_map[i]], sizeof(float) * number_map[i], cudaMemcpyDeviceToHost);
	}
}

float Convolutional_Neural_Networks_CUDA::Load_Parameter(char path[]){
	float epsilon = 0;

	FILE *file = fopen(path, "rt");

	if(file){
		fscanf(file, "%f", &epsilon);

		for(int h = 0;h < number_parameter_type;h++){
			for(int i = 0;i < number_layer;i++){
				if(Access_Parameter(h, i)){
					if(strstr(type_layer[i], "bn")){
						for(int j = 0;j < number_map[i];j++){
							fscanf(file, "%f", &gamma[h][i][j]);
							fscanf(file, "%f", &beta[h][i][j]);
						}
					}

					int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
					int number_parameter	= number_map[i] * (number_map[lower_layer_index] + 1) * length_filter[i] * length_filter[i];

					for(int j = 0;j < number_parameter;j++){
						fscanf(file, "%f", &weight[h][i][j]);
					}
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load_Parameter], %s not found.\n", path);
	}
	return epsilon;
}
float Convolutional_Neural_Networks_CUDA::Train(int batch_size, int number_train, float epsilon, float learning_rate, float **input, float **target_output){
	int *index = new int[number_train];

	float loss = 0;

	float *target_output_batch;
	float *temporal_loss;

	for(int i = 0;i < number_train;i++){
		index[i] = i;
	}
	for(int i = 0;i < number_train;i++){
		int j = rand() % number_train;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;
	}

	cudaMalloc(&target_output_batch, sizeof(float) * batch_size * number_map[number_layer - 1]);
	cudaMalloc(&temporal_loss,		 sizeof(float));
	cudaMemset(temporal_loss, 0, sizeof(float));
	Resize_Memory(batch_size);

	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				cudaMemset(sum_mean[h][i],		0, sizeof(float) * number_map[i]);
				cudaMemset(sum_variance[h][i],	0, sizeof(float) * number_map[i]);
			}
		}
	}
	this->epsilon = epsilon;

	for(int g = 0, h = 0;g < number_train;g++){
		cudaMemcpy(&neuron[0][0][h * number_map[0] * length_map[0] * length_map[0]], input[index[g]], sizeof(float) * number_map[0] * length_map[0] * length_map[0], cudaMemcpyHostToDevice);
		cudaMemcpy(&target_output_batch[h * number_map[number_layer - 1]], target_output[index[g]], sizeof(float) * number_map[number_layer - 1], cudaMemcpyHostToDevice);

		if(++h == batch_size){
			h = 0;

			for(int i = 1;i < number_layer;i++){
				Feedforward (i);
				Activate	("train", i);
			}

			for(int i = number_layer - 1;i > 0;i--){
				Backpropagate(i);
				Differentiate(i, learning_rate, target_output_batch);
			}
			for(int i = number_layer - 1;i > 0;i--){
				Adjust_Parameter(i);
			}

			int i = number_layer - 1;

			if(strstr(type_layer[i], "ce")){
				Calculate_Loss<<<1, NUMBER_THREAD>>>(0, batch_size * number_map[i], temporal_loss, neuron[0][i], target_output_batch);
			}
			else
			if(strstr(type_layer[i], "mse")){
				Calculate_Loss<<<1, NUMBER_THREAD>>>(1, batch_size * number_map[i], temporal_loss, neuron[0][i], target_output_batch);
			}
		}
	}

	for(int h = 0;h < number_parameter_type;h++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				Multiply<<<number_map[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_map[i], sum_mean[h][i],		(double)batch_size / number_train, mean[h][i]);
				Multiply<<<number_map[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_map[i], sum_variance[h][i],	(double)batch_size / (batch_size - 1) * batch_size / number_train, variance[h][i]);
			}
		}
	}

	cudaMemcpy(&loss, temporal_loss, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(target_output_batch);
	cudaFree(temporal_loss);
	delete[] index;

	return loss / number_train;
}