#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "CNN.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMBER_THREAD 64 // must be a power of 2

__global__ void ::Activate(int option, int number_neurons, float neuron[]){
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j < number_neurons){
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
__global__ void ::Adjust_Parameter(int batch_size, int lower_layer_index, int layer_index, int kernel_width, int kernel_height, int stride_width, int stride_height, int map_width[], int map_height[], int number_maps[], float derivative[], float lower_neuron[], float weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int lower_i	= lower_layer_index;

	int j = index / ((number_maps[lower_i] + 1) * kernel_height * kernel_width);

	if(j < number_maps[i]){
		int m = ((index % ((number_maps[lower_i] + 1) * kernel_height * kernel_width)) / (kernel_height * kernel_width));
		int n = ((index % ((number_maps[lower_i] + 1) * kernel_height * kernel_width)) % (kernel_height * kernel_width)) / kernel_width;
		int o = ((index % ((number_maps[lower_i] + 1) * kernel_height * kernel_width)) % (kernel_height * kernel_width)) % kernel_width;

		if(m < number_maps[lower_i]){
			float sum = 0;

			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						int index[2] = {k * stride_height + n, l * stride_width + o};

						if(index[0] < map_height[lower_i] && index[1] < map_width[lower_i]){
							sum += derivative[h * number_maps[i] * map_height[i] * map_width[i] +
											  j * map_height[i] * map_width[i] +
											  k * map_width[i] +
											  l]
								* lower_neuron[h * number_maps[lower_i] * map_height[lower_i] * map_width[lower_i] +
											   m * map_height[lower_i] * map_width[lower_i] +
											   index[0] * map_width[lower_i] +
											   index[1]];
						}
					}
				}
			}
			weight[index] -= sum;
		}
		else
		if(m == number_maps[lower_i] && n == 0 && o == 0){
			float sum = 0;

			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						sum += derivative[h * number_maps[i] * map_height[i] * map_width[i] +
										  j * map_height[i] * map_width[i] +
										  k * map_width[i] +
										  l];
					}
				}
			}
			weight[index] -= sum;
		}
	}
}
__global__ void ::Backpropagate(int option, int batch_size, int layer_index, int upper_layer_index, int upper_kernel_width, int upper_kernel_height, int upper_stride_width, int upper_stride_height, int map_width[], int map_height[], int number_maps[], float derivative[], float upper_derivative[], float upper_weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int upper_i	= upper_layer_index;

	int h = index / (number_maps[i] * map_height[i] * map_width[i]);

	if(h < batch_size){
		int j = ((index % (number_maps[i] * map_height[i] * map_width[i])) / (map_height[i] * map_width[i]));
		int k = ((index % (number_maps[i] * map_height[i] * map_width[i])) % (map_height[i] * map_width[i])) / map_width[i];
		int l = ((index % (number_maps[i] * map_height[i] * map_width[i])) % (map_height[i] * map_width[i])) % map_width[i];

		if(option == 0){
			int ks				= k / upper_stride_height;
			int ls				= l / upper_stride_width;
			int neuron_index[2] = {ks - (upper_kernel_height - 1), ls - (upper_kernel_width - 1)};

			float sum = 0;

			if(neuron_index[0] < 0) neuron_index[0] = 0;
			if(neuron_index[1] < 0) neuron_index[1] = 0;

			for(int m = 0;m < number_maps[upper_i];m++){
				for(int n = neuron_index[0];n < map_height[upper_i] && n <= ks;n++){
					for(int o = neuron_index[1];o < map_width[upper_i] && o <= ls;o++){
						sum += upper_derivative[h * number_maps[upper_i] * map_height[upper_i] * map_width[upper_i] +
												m * map_height[upper_i] * map_width[upper_i] +
												n * map_width[upper_i] +
												o]
							* upper_weight[m * (number_maps[i] + 1) * upper_kernel_height * upper_kernel_width +
										   j * upper_kernel_height * upper_kernel_width +
										   abs(ks - n) * upper_kernel_width +
										   abs(ls - o)];
					}
				}
			}
			derivative[index] = sum;
		}
		else
		if(option == 1){
			int stride[] = {map_height[i] / map_height[i + 1], map_width[i] / map_width[i + 1]};

			derivative[index] = upper_derivative[h * number_maps[i + 1] * map_height[i + 1] * map_width[i + 1] +
												 j * map_height[i + 1] * map_width[i + 1] +
												 (k / stride[0]) * map_width[i + 1] +
												 (l / stride[1])];
		}
		else
		if(option == 2){
			int margin[] = {(map_height[i + 1] - map_height[i]) / 2, (map_width[i + 1] - map_width[i]) / 2};

			derivative[index] = upper_derivative[h * number_maps[i + 1] * map_height[i + 1] * map_width[i + 1] +
												 j * map_height[i + 1] * map_width[i + 1] +
												 (margin[0] + k) * map_width[i + 1] +
												 (margin[1] + l)];
		}
	}
}
__global__ void ::Batch_Normalization_Activate(int option, int batch_size, int map_width, int map_height, int number_maps, float epsilon, float gamma[], float beta[], float mean[], float variance[], float sum_mean[], float sum_variance[], float neuron[], float neuron_batch_0[], float neuron_batch_1[]){
	int j = blockIdx.x;

	if(option == 0){
		__shared__ float sum[NUMBER_THREAD];

		sum[threadIdx.x] = 0;
		for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
			int h = m / (map_height * map_width);
			int k = m % (map_height * map_width);

			sum[threadIdx.x] += neuron[h * number_maps * map_height * map_width +
									   j * map_height * map_width +
									   k];
		}
		for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
			__syncthreads();

			if(threadIdx.x < m){
				sum[threadIdx.x] += sum[threadIdx.x + m];
			}
		}
		if(threadIdx.x == 0){
			sum_mean[j] += (mean[j] = sum[0] / (batch_size * map_height * map_width));
		}
		__syncthreads();

		sum[threadIdx.x] = 0;
		for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
			int h = m / (map_height * map_width);
			int k = m % (map_height * map_width);

			int index = h * number_maps * map_height * map_width +
						j * map_height * map_width +
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
			sum_variance[j] += (variance[j] = sum[0] / (batch_size * map_height * map_width));
		}
		__syncthreads();

		for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
			int h = m / (map_height * map_width);
			int k = m % (map_height * map_width);

			int index = h * number_maps * map_height * map_width +
						j * map_height * map_width +
						k;

			neuron_batch_0[index]	= (neuron[index] - mean[j]) / sqrt(variance[j] + epsilon);
			neuron_batch_1[index]	= neuron[index];
			neuron[index]			= gamma[j] * neuron_batch_0[index] + beta[j];
		}
	}
	else
	if(option == 1){
		for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
			int h = m / (map_height * map_width);
			int k = m % (map_height * map_width);

			int index = h * number_maps * map_height * map_width +
						j * map_height * map_width +
						k;

			float stdv = sqrt(variance[j] + epsilon);

			neuron[index] = gamma[j] / stdv * neuron[index] + (beta[j] - gamma[j] * mean[j] / stdv);
		}
	}
}
__global__ void ::Batch_Normalization_Adjust_Parameter(int batch_size, int map_width, int map_height, int number_maps, float gamma[], float beta[], float derivative_batch_1[], float neuron_batch_0[]){
	int j = blockIdx.x;

	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
		int h = m / (map_height * map_width);
		int k = m % (map_height * map_width);

		int index = h * number_maps * map_height * map_width +
					j * map_height * map_width +
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
		gamma[j] -= sum[0];
	}

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
		int h = m / (map_height * map_width);
		int k = m % (map_height * map_width);

		sum[threadIdx.x] += derivative_batch_1[h * number_maps * map_height * map_width +
											   j * map_height * map_width +
											   k];
	}
	for(int m = (blockDim.x >> 1);m;m = (m >> 1)){
		__syncthreads();

		if(threadIdx.x < m){
			sum[threadIdx.x] += sum[threadIdx.x + m];
		}
	}
	if(threadIdx.x == 0){
		beta[j] -= sum[0];
	}
}
__global__ void ::Batch_Normalization_Differentiate(int batch_size, int map_width, int map_height, int number_maps, float epsilon, float gamma[], float beta[], float mean[], float variance[], float derivative[], float derivative_batch_0[], float derivative_batch_1[], float neuron_batch_1[]){
	int j = blockIdx.x;

	__shared__ float derivative_mean;
	__shared__ float derivative_variance;
	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
		int h = m / (map_height * map_width);
		int k = m % (map_height * map_width);

		int index = h * number_maps * map_height * map_width +
					j * map_height * map_width +
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
	for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
		int h = m / (map_height * map_width);
		int k = m % (map_height * map_width);

		sum[threadIdx.x] += derivative_batch_0[h * number_maps * map_height * map_width + j * map_height * map_width + k];
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

	for(int m = threadIdx.x;m < batch_size * map_height * map_width;m += blockDim.x){
		int h = m / (map_height * map_width);
		int k = m % (map_height * map_width);

		int index = h * number_maps * map_height * map_width + j * map_height * map_width + k;

		derivative_batch_1[index]	= derivative[index];
		derivative[index]			= derivative_batch_0[index] / sqrt(variance[j] + epsilon) + derivative_variance * 2 * (neuron_batch_1[index] - mean[j]) / (batch_size * map_height * map_width) + derivative_mean / (batch_size * map_height * map_width);
	}
}
__global__ void ::Calculate_Loss(int option, int number_neurons, float *loss, float neuron[], float target_output[]){
	__shared__ float sum[NUMBER_THREAD];

	sum[threadIdx.x] = 0;
	for(int j = threadIdx.x;j < number_neurons;j += blockDim.x){
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
__global__ void ::Dropout(int option, int batch_size, int number_maps, int map_width, int map_height, int seed, float rate, float neuron[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int h = (index / (number_maps * map_height * map_width));
	int j = (index % (number_maps * map_height * map_width)) / (map_height * map_width);

	if(j < number_maps){
		curandState s[NUMBER_THREAD];

		if(option == 0){
			curand_init(seed + h * number_maps * map_height * map_width + j, 0, 0, &s[threadIdx.x]);

			neuron[index] *= (curand_uniform(&s[threadIdx.x]) <= rate);
		}
		else
		if(option == 1){
			neuron[index] *= rate;
		}
	}
}
__global__ void ::Feedforward(int option, int batch_size, int lower_layer_index, int layer_index, int kernel_width, int kernel_height, int stride_width, int stride_height, int number_maps[], int map_width[], int map_height[], float lower_neuron[], float neuron[], float weight[]){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i		= layer_index;
	int lower_i	= lower_layer_index;

	int h = index / (number_maps[i] * map_height[i] * map_width[i]);

	if(h < batch_size){
		int j = ((index % (number_maps[i] * map_height[i] * map_width[i])) / (map_height[i] * map_width[i]));
		int k = ((index % (number_maps[i] * map_height[i] * map_width[i])) % (map_height[i] * map_width[i])) / map_width[i];
		int l = ((index % (number_maps[i] * map_height[i] * map_width[i])) % (map_height[i] * map_width[i])) % map_width[i];

		if(option == 0){
			float sum = 0;

			for(int m = 0;m < number_maps[lower_i];m++){
				for(int n = 0;n < kernel_height;n++){
					for(int o = 0;o < kernel_width;o++){
						int neuron_index[2] = {k * stride_height + n, l * stride_width + o};

						if(neuron_index[0] < map_height[lower_i] && neuron_index[1] < map_width[lower_i]){
							sum += lower_neuron[h * number_maps[lower_i] * map_height[lower_i] * map_width[lower_i] +
												m * map_height[lower_i] * map_width[lower_i] +
												neuron_index[0] * map_width[lower_i] +
												neuron_index[1]]
								* weight[j * (number_maps[lower_i] + 1) * kernel_height * kernel_width +
										 m * kernel_height * kernel_width +
										 n * kernel_width +
										 o];
						}
					}
				}
			}
			neuron[index] = sum + weight[j * (number_maps[lower_i] + 1) * kernel_height * kernel_width +
										 number_maps[lower_i] * kernel_height * kernel_width];
		}
		else
		if(option == 1){
			int stride[] = {map_height[i - 1] / map_height[i], map_width[i - 1] / map_width[i]};

			float sum = 0;
						
			for(int m = 0;m < stride[0];m++){
				for(int n = 0;n < stride[1];n++){
					sum += lower_neuron[h * number_maps[i - 1] * map_height[i - 1] * map_width[i - 1] +
										j * map_height[i - 1] * map_width[i - 1] +
										(k * stride[0] + m) * map_width[i - 1] +
										(l * stride[1] + n)];
				}
			}
			neuron[index] = sum / (stride[0] * stride[1]);
		}
		else
		if(option == 2){
			int stride[] = {map_height[i - 1] / map_height[i], map_width[i - 1] / map_width[i]};

			float max = -1;

			for(int m = 0;m < stride[0];m++){
				for(int n = 0;n < stride[1];n++){
					int neuron_index = h * number_maps[i - 1] * map_height[i - 1] * map_width[i - 1] +
									   j * map_height[i - 1] * map_width[i - 1] +
									   (k * stride[0] + m) * map_width[i - 1] +
									   (l * stride[1] + n);

					if(max < lower_neuron[neuron_index]){
						max = lower_neuron[neuron_index];
					}
				}
			}
			neuron[index] = max;
		}
		else
		if(option == 3){
			int margin[] = {(map_height[i] - map_height[i - 1]) / 2, (map_width[i] - map_width[i - 1]) / 2};

			if(k < map_height[i - 1] && l < map_width[i - 1]){
				neuron[h * number_maps[i] * map_height[i] * map_width[i] +
					   j * map_height[i] * map_width[i] +
					   (margin[0] + k) * map_width[i] +
					   (margin[1] + l)]
				= lower_neuron[h * number_maps[i - 1] * map_height[i - 1] * map_width[i - 1] +
							   j * map_height[i - 1] * map_width[i - 1] +
							   k * map_width[i - 1] +
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
__global__ void ::Softmax(int batch_size, int map_width, int map_height, int number_maps, float neuron[]){
	int h = blockIdx.x * blockDim.x + threadIdx.x;

	if(h < batch_size){
		float max = 0;
		float sum = 0;

		for(int j = 0;j < number_maps;j++){
			int index = h * number_maps * map_height * map_width +
						j * map_height * map_width;

			if(max < neuron[index]){
				max = neuron[index];
			}
		}
		for(int j = 0;j < number_maps;j++){
			int index = h * number_maps * map_height * map_width +
						j * map_height * map_width;

			sum += (neuron[index] = exp(neuron[index] - max));
		}
		for(int j = 0;j < number_maps;j++){
			neuron[h * number_maps * map_height * map_width +
				   j * map_height * map_width] /= sum;
		}
	}
}

void Convolutional_Neural_Networks_CUDA::Activate(char option[], int layer_index){
	int i = layer_index;

	int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

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
				::Dropout<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, number_maps[i], map_width[i], map_height[i], clock(), atof(rate), neuron);
			}
			else
			if(!strcmp(option, "test")){
				::Dropout<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, number_maps[i], map_width[i], map_height[i], clock(), atof(rate), neuron);
			}
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "ce")){
			if(strstr(type_layer[i], "sm")){
				::Softmax<<<batch_size / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, map_width[i], map_height[i], number_maps[i], neuron);
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

	int number_parameters = number_maps[i] * (number_maps[i - 1] + 1) * kernel_height[i] * kernel_width[i];

	float *derivative	= this->derivative[0][i];
	float *lower_neuron	= this->neuron[0][i - 1];
	float *weight		= this->weight[0][i];

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Adjust_Parameter("normal", layer_index);
		}
		::Adjust_Parameter<<<number_parameters / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, i - 1, layer_index, kernel_width[i], kernel_height[i], stride_width[i], stride_height[i], map_width_factor, map_height_factor, number_maps_factor, derivative, lower_neuron, weight);

		if(strstr(type_layer[i], "psc")){
			int lower_layer_index = i - atoi(strstr(type_layer[i], "psc") + 2);

			lower_neuron	= this->neuron[0][lower_layer_index];
			derivative		= this->derivative[1][i];
			weight			= this->weight[1][i];

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Adjust_Parameter("shortcut", layer_index);
			}
			::Adjust_Parameter<<<number_parameters / NUMBER_THREAD + 1, NUMBER_THREAD>>>(batch_size, lower_layer_index, layer_index, shortcut_kernel_width[i], shortcut_kernel_height[i], shortcut_stride_width[i], shortcut_stride_height[i], map_width_factor, map_height_factor, number_maps_factor, derivative, lower_neuron, weight);
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Backpropagate(int layer_index){
	if(layer_index == number_layers - 1){
		return;
	}

	int i = layer_index;

	int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

	float *derivative		= this->derivative[0][i];
	float *upper_derivative	= this->derivative[0][i + 1];
	float *upper_weight		= this->weight[0][i + 1];

	if(type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L'){
		::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, layer_index, i + 1, kernel_width[i + 1], kernel_height[i + 1], stride_width[i + 1], stride_height[i + 1], map_width_factor, map_height_factor, number_maps_factor, derivative, upper_derivative, upper_weight);
	}
	else
	if(type_layer[i + 1][0] == 'P'){
		if(strstr(type_layer[i + 1], "avg") || strstr(type_layer[i + 1], "max")){
			::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, layer_index, i + 1, kernel_width[i + 1], kernel_height[i + 1], stride_width[i + 1], stride_height[i + 1], map_width_factor, map_height_factor, number_maps_factor, derivative, upper_derivative, upper_weight);
		}
		else
		if(strstr(type_layer[i + 1], "pad")){
			::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, batch_size, layer_index, i + 1, kernel_width[i + 1], kernel_height[i + 1], stride_width[i + 1], stride_height[i + 1], map_width_factor, map_height_factor, number_maps_factor, derivative, upper_derivative, upper_weight);
		}
	}

	if(strstr(type_layer[i], "sc")){
		int upper_layer_index = i;

		for(int upper_i = i + 1;upper_i < number_layers;upper_i++){
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
				::Backpropagate<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, layer_index, upper_layer_index, shortcut_kernel_width[upper_layer_index],shortcut_kernel_height[upper_layer_index], shortcut_stride_width[upper_layer_index], shortcut_stride_height[upper_layer_index], map_width_factor, map_height_factor, number_maps_factor, derivative, upper_derivative, upper_weight);
			}
			else{
				cudaMemcpy(derivative, upper_derivative, sizeof(float) * number_memory, cudaMemcpyDeviceToDevice);
			}
			::Add<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_memory, derivative, this->derivative[0][i], this->derivative[0][i]);
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Differentiate(int layer_index, float learning_rate, float target_output[]){
	int i = layer_index;

	int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

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
				// derivative = derivative;
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

	int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

	float *lower_neuron = this->neuron[0][i - 1];
	float *neuron		= this->neuron[0][i];
	float *weight		= this->weight[0][i];

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, i - 1, layer_index, kernel_width[i], kernel_height[i], stride_width[i], stride_height[i], number_maps_factor, map_width_factor, map_height_factor, lower_neuron, neuron, weight);
	}
	else
	if(type_layer[i][0] == 'P'){
		if(strstr(type_layer[i], "avg")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(1, batch_size, i - 1, layer_index, kernel_width[i], kernel_height[i], stride_width[i], stride_height[i], number_maps_factor, map_width_factor, map_height_factor, lower_neuron, neuron, weight);
		}
		else
		if(strstr(type_layer[i], "max")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(2, batch_size, i - 1, layer_index, kernel_width[i], kernel_height[i], stride_width[i], stride_height[i], number_maps_factor, map_width_factor, map_height_factor, lower_neuron, neuron, weight);
		}
		else
		if(strstr(type_layer[i], "pad")){
			::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(3, batch_size, i - 1, layer_index, kernel_width[i], kernel_height[i], stride_width[i], stride_height[i], number_maps_factor, map_width_factor, map_height_factor, lower_neuron, neuron, weight);
		}
	}

	if(strstr(type_layer[i], "sc")){
		int lower_layer_index = i - atoi(strstr(type_layer[i], "sc") + 2);

		if(lower_layer_index < i){
			lower_neuron = this->neuron[0][lower_layer_index];
			neuron		 = this->neuron[1][i];
			weight		 = this->weight[1][i];

			if(strstr(type_layer[i], "psc")){
				::Feedforward<<<number_memory / NUMBER_THREAD + 1, NUMBER_THREAD>>>(0, batch_size, lower_layer_index, layer_index, shortcut_kernel_width[i], shortcut_kernel_height[i], shortcut_stride_width[i], shortcut_stride_height[i], number_maps_factor, map_width_factor, map_height_factor, lower_neuron, neuron, weight);
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
		::Batch_Normalization_Activate<<<number_maps[i], NUMBER_THREAD>>>(0, batch_size, map_width[i], map_height[i], number_maps[i], epsilon, gamma, beta, mean, variance, sum_mean, sum_variance, neuron, neuron_batch[0], neuron_batch[1]);
	}
	else
	if(!strcmp(option, "test")){
		::Batch_Normalization_Activate<<<number_maps[i], NUMBER_THREAD>>>(1, batch_size, map_width[i], map_height[i], number_maps[i], epsilon, gamma, beta, mean, variance, sum_mean, sum_variance, neuron, neuron_batch[0], neuron_batch[1]);
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
	::Batch_Normalization_Adjust_Parameter<<<number_maps[i], NUMBER_THREAD>>>(batch_size, map_width[i], map_height[i], number_maps[i], gamma, beta, derivative_batch, neuron_batch);
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
	::Batch_Normalization_Differentiate<<<number_maps[i], NUMBER_THREAD>>>(batch_size, map_width[i], map_height[i], number_maps[i], epsilon, gamma, beta, mean, variance, derivative, derivative_batch[0], derivative_batch[1], neuron_batch[1]);
}

void Convolutional_Neural_Networks_CUDA::Resize_Memory(int batch_size){
	if(this->batch_size != batch_size){
		for(int g = 0;g < number_memory_types;g++){
			for(int i = 0;i < number_layers;i++){
				if(Access_Memory(g, i)){
					int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

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

Convolutional_Neural_Networks_CUDA::Convolutional_Neural_Networks_CUDA(char **type_layer, int number_layers, int map_width[], int map_height[], int number_maps[]){
	this->kernel_width	= new int[number_layers];
	this->kernel_height	= new int[number_layers];
	this->map_width		= new int[number_layers];
	this->map_height	= new int[number_layers];
	this->number_layers	= number_layers;
	this->number_maps	= new int[number_layers];
	this->stride_width	= new int[number_layers];
	this->stride_height	= new int[number_layers];
	this->type_layer	= new char*[number_layers];

	shortcut_kernel_width	= new int[number_layers];
	shortcut_kernel_height	= new int[number_layers];
	shortcut_stride_width	= new int[number_layers];
	shortcut_stride_height	= new int[number_layers];

	cudaMalloc(&map_width_factor, sizeof(int) * number_layers);
	cudaMalloc(&map_height_factor, sizeof(int) * number_layers);
	cudaMalloc(&number_maps_factor, sizeof(int) * number_layers);

	batch_size				= 1;
	number_memory_types		= 6;
	number_parameter_types	= 2;

	for(int i = 0;i < number_layers;i++){
		this->type_layer[i]	 = new char[strlen(type_layer[i]) + 1];
		strcpy(this->type_layer[i], type_layer[i]);
		this->number_maps[i] = number_maps[i];
		cudaMemcpy(&number_maps_factor[i], &(this->number_maps[i]), sizeof(int), cudaMemcpyHostToDevice);
		this->map_width[i]	 = (map_width == 0) ? (1):(map_width[i]);
		cudaMemcpy(&map_width_factor[i], &(this->map_width[i]), sizeof(int), cudaMemcpyHostToDevice);
		this->map_height[i]	 = (map_height == 0) ? (1):(map_height[i]);
		cudaMemcpy(&map_height_factor[i], &(this->map_height[i]), sizeof(int), cudaMemcpyHostToDevice);
	}
	for(int i = 1;i < number_layers;i++){
		char *type = strtok(this->type_layer[i], "/");

		if(strstr(type, "ks")){
			char *kernel_size = strstr(type, "ks");

			kernel_width[i] = atoi(kernel_size + 2);
			kernel_size = strstr(kernel_size, ",");
			kernel_height[i] = (kernel_size && atoi(kernel_size + 1) > 0) ? (atoi(kernel_size + 1)):(kernel_width[i]);
		}
		else{
			kernel_width[i]	 = (i == 0 || type_layer[i][0] == 'P') ? (0):(this->map_width[i - 1] - this->map_width[i] + 1);
			kernel_height[i] = (i == 0 || type_layer[i][0] == 'P') ? (0):(this->map_height[i - 1] - this->map_height[i] + 1);
		}

		if(strstr(type, "st")){
			char *stride = strstr(type, "st");

			stride_width[i] = atoi(stride + 2);
			stride = strstr(stride, ",");
			stride_height[i] = (stride && atoi(stride + 1) > 0) ? (atoi(stride + 1)):(stride_width[i]);
		}
		else{
			stride_width[i]	 = 1;
			stride_height[i] = 1;
		}

		strcpy(this->type_layer[i], type_layer[i]);

		if(strstr(type_layer[i], "psc")){
			char *type_shortcut = strstr(type_layer[i], "psc");

			shortcut_kernel_width[i]	= 1;
			shortcut_kernel_height[i]	= 1;

			if(strstr(type_shortcut, "st")){
				char *stride = strstr(type_shortcut, "st");

				shortcut_stride_width[i] = atoi(stride + 2);
				stride = strstr(stride, ",");
				shortcut_stride_height[i] = (stride && atoi(stride + 1) > 0) ? (atoi(stride + 1)):(shortcut_stride_width[i]);
			}
			else{
				shortcut_stride_width[i]	= 1;
				shortcut_stride_height[i]	= 1;
			}
		}
	}

	gamma		 = new float**[number_parameter_types];
	beta		 = new float**[number_parameter_types];
	mean		 = new float**[number_parameter_types];
	variance	 = new float**[number_parameter_types];
	sum_mean	 = new float**[number_parameter_types];
	sum_variance = new float**[number_parameter_types];

	for(int h = 0;h < number_parameter_types;h++){
		gamma[h]		= new float*[number_layers];
		beta[h]			= new float*[number_layers];
		mean[h]			= new float*[number_layers];
		variance[h]		= new float*[number_layers];
		sum_mean[h]		= new float*[number_layers];
		sum_variance[h]	= new float*[number_layers];

		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				cudaMalloc(&gamma[h][i],		sizeof(float) * number_maps[i]);
				cudaMalloc(&beta[h][i],			sizeof(float) * number_maps[i]);
				cudaMalloc(&mean[h][i],			sizeof(float) * number_maps[i]);
				cudaMalloc(&variance[h][i],		sizeof(float) * number_maps[i]);
				cudaMalloc(&sum_mean[h][i],		sizeof(float) * number_maps[i]);
				cudaMalloc(&sum_variance[h][i],	sizeof(float) * number_maps[i]);
			}
		}
	}

	derivative	= new float**[number_memory_types];
	neuron		= new float**[number_memory_types];

	for(int g = 0;g < number_memory_types;g++){
		derivative[g]	= new float*[number_layers];
		neuron[g]		= new float*[number_layers];

		for(int i = 0;i < number_layers;i++){
			if(Access_Memory(g, i)){
				int number_memory = batch_size * number_maps[i] * map_height[i] * map_width[i];

				cudaMalloc(&derivative[g][i],	sizeof(float) * number_memory);
				cudaMalloc(&neuron[g][i],		sizeof(float) * number_memory);
			}
		}
	}

	weight = new float**[number_parameter_types];

	for(int h = 0;h < number_parameter_types;h++){
		weight[h] = new float*[number_layers];

		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i)){
				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "psc") + 2)):(i - 1);
				int number_parameters	= number_maps[i] * (number_maps[lower_layer_index] + 1) * kernel_height[i] * kernel_width[i];

				if(number_parameters / NUMBER_THREAD + 1 > 65535){
					fprintf(stderr, "[required gridDim: %d > 65535], (NUMBER_THREAD: %d) must be a higher value.\nplease refer to the CNN.cu/line 11\n", number_parameters / NUMBER_THREAD + 1, NUMBER_THREAD);
				}
				cudaMalloc(&weight[h][i], sizeof(float) * number_parameters);
			}
		}
	}
}
Convolutional_Neural_Networks_CUDA::~Convolutional_Neural_Networks_CUDA(){
	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
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

	for(int g = 0;g < number_memory_types;g++){
		for(int i = 0;i < number_layers;i++){
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

	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i)){
				cudaFree(weight[h][i]);
			}
		}
		delete[] weight[h];
	}
	delete[] weight;

	for(int i = 0;i < number_layers;i++){
		delete[] type_layer[i];
	}
	delete[] type_layer;

	delete[] kernel_width;
	delete[] kernel_height;
	delete[] map_width;
	delete[] map_height;
	delete[] number_maps;
	delete[] stride_width;
	delete[] stride_height;

	delete[] shortcut_kernel_width;
	delete[] shortcut_kernel_height;
	delete[] shortcut_stride_width;
	delete[] shortcut_stride_height;

	cudaFree(map_width_factor);
	cudaFree(map_height_factor);
	cudaFree(number_maps_factor);
}

void Convolutional_Neural_Networks_CUDA::Initialize_Parameter(int seed, double scale, double shift){
	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i)){
				if(strstr(type_layer[i], "bn")){
					::Set<<<number_maps[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_maps[i], 1, gamma[h][i]);
					::Set<<<number_maps[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_maps[i], 0, beta[h][i]);
				}

				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
				int number_parameters	= number_maps[i] * (number_maps[lower_layer_index] + 1) * kernel_height[i] * kernel_width[i];

				::Randomize<<<number_parameters / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_parameters, seed, scale, shift, weight[h][i]);
			}
		}
	}
}
void Convolutional_Neural_Networks_CUDA::Load_Parameter(char path[]){
	FILE *file = fopen(path, "rt");

	if(file){
		fscanf(file, "%f", &epsilon);

		for(int h = 0;h < number_parameter_types;h++){
			for(int i = 0;i < number_layers;i++){
				if(Access_Parameter(h, i)){
					float *parameter;

					if(strstr(type_layer[i], "bn")){
						float *memory = new float[number_maps[i]];

						for(int j = 0;j < number_maps[i];j++) fscanf(file, "%f", &memory[j]);
						cudaMemcpy(gamma[h][i], memory,		sizeof(float) * number_maps[i], cudaMemcpyHostToDevice);
						for(int j = 0;j < number_maps[i];j++) fscanf(file, "%f", &memory[j]);
						cudaMemcpy(beta[h][i], memory,		sizeof(float) * number_maps[i], cudaMemcpyHostToDevice);
						for(int j = 0;j < number_maps[i];j++) fscanf(file, "%f", &memory[j]);
						cudaMemcpy(mean[h][i], memory,		sizeof(float) * number_maps[i], cudaMemcpyHostToDevice);
						for(int j = 0;j < number_maps[i];j++) fscanf(file, "%f", &memory[j]);
						cudaMemcpy(variance[h][i], memory,	sizeof(float) * number_maps[i], cudaMemcpyHostToDevice);

						delete[] memory;
					}

					int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
					int number_parameters	= number_maps[i] * (number_maps[lower_layer_index] + 1) * kernel_height[i] * kernel_width[i];

					parameter = new float[number_parameters];

					for(int j = 0;j < number_parameters;j++)	fscanf(file, "%f", &parameter[j]);
					cudaMemcpy(weight[h][i], parameter, sizeof(float) * number_parameters, cudaMemcpyHostToDevice);

					delete[] parameter;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load_Parameter], %s not found\n", path);
	}
}
void Convolutional_Neural_Networks_CUDA::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	fprintf(file, "%f\n", epsilon);

	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i)){
				float *parameter;

				if(strstr(type_layer[i], "bn")){
					float *memory = new float[number_maps[i]];

					cudaMemcpy(memory, gamma[h][i],		sizeof(float) * number_maps[i], cudaMemcpyDeviceToHost);
					for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", memory[j]);
					cudaMemcpy(memory, beta[h][i],		sizeof(float) * number_maps[i], cudaMemcpyDeviceToHost);
					for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", memory[j]);
					cudaMemcpy(memory, mean[h][i],		sizeof(float) * number_maps[i], cudaMemcpyDeviceToHost);
					for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", memory[j]);
					cudaMemcpy(memory, variance[h][i],	sizeof(float) * number_maps[i], cudaMemcpyDeviceToHost);
					for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", memory[j]);

					delete[] memory;
				}

				int lower_layer_index	= (h == 1) ? (i - atoi(strstr(type_layer[i], "sc") + 2)):(i - 1);
				int number_parameters	= number_maps[i] * (number_maps[lower_layer_index] + 1) * kernel_height[i] * kernel_width[i];

				cudaMemcpy(parameter = new float[number_parameters], weight[h][i], sizeof(float) * number_parameters, cudaMemcpyDeviceToHost);
				for(int j = 0;j < number_parameters;j++) fprintf(file, "%f\n", parameter[j]);

				delete[] parameter;
			}
		}
	}
	fclose(file);
}
void Convolutional_Neural_Networks_CUDA::Test(float input[], float output[]){
	Resize_Memory(1);

	cudaMemcpy(neuron[0][0], input, sizeof(float) * number_maps[0] * map_height[0] * map_width[0], cudaMemcpyHostToDevice);

	for(int i = 1;i < number_layers;i++){
		Feedforward	(i);
		Activate	("test", i);
	}
	cudaMemcpy(output, neuron[0][number_layers - 1], sizeof(float) * number_maps[number_layers - 1], cudaMemcpyDeviceToHost);
}
void Convolutional_Neural_Networks_CUDA::Test(int batch_size, float **input, float **output){
	Resize_Memory(batch_size);

	for(int h = 0, i = 0;h < batch_size;h++){
		cudaMemcpy(&neuron[0][i][h * number_maps[i] * map_height[i] * map_width[i]], input[h], sizeof(float) * number_maps[i] * map_height[i] * map_width[i], cudaMemcpyHostToDevice);
	}
	for(int i = 1;i < number_layers;i++){
		Feedforward	(i);
		Activate	("test", i);
	}
	for(int h = 0, i = number_layers - 1;h < batch_size;h++){
		cudaMemcpy(output[h], &neuron[0][i][h * number_maps[i]], sizeof(float) * number_maps[i], cudaMemcpyDeviceToHost);
	}
}

float Convolutional_Neural_Networks_CUDA::Train(int batch_size, int number_training, float epsilon, float learning_rate, float **input, float **target_output){
	int *index = new int[number_training];

	float loss = 0;

	float *target_output_batch;
	float *temporal_loss;

	for(int i = 0;i < number_training;i++){
		index[i] = i;
	}
	for(int i = 0;i < number_training;i++){
		int j = rand() % number_training;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;
	}

	cudaMalloc(&target_output_batch, sizeof(float) * batch_size * number_maps[number_layers - 1]);
	cudaMalloc(&temporal_loss,		 sizeof(float));
	cudaMemset(temporal_loss, 0, sizeof(float));
	Resize_Memory(batch_size);

	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				cudaMemset(sum_mean[h][i],		0, sizeof(float) * number_maps[i]);
				cudaMemset(sum_variance[h][i],	0, sizeof(float) * number_maps[i]);
			}
		}
	}
	this->epsilon = epsilon;

	for(int g = 0, h = 0;g < number_training;g++){
		cudaMemcpy(&neuron[0][0][h * number_maps[0] * map_height[0] * map_width[0]], input[index[g]], sizeof(float) * number_maps[0] * map_height[0] * map_width[0], cudaMemcpyHostToDevice);
		cudaMemcpy(&target_output_batch[h * number_maps[number_layers - 1]], target_output[index[g]], sizeof(float) * number_maps[number_layers - 1], cudaMemcpyHostToDevice);

		if(++h == batch_size){
			h = 0;

			for(int i = 1;i < number_layers;i++){
				Feedforward (i);
				Activate	("train", i);
			}

			for(int i = number_layers - 1;i > 0;i--){
				Backpropagate(i);
				Differentiate(i, learning_rate, target_output_batch);
			}
			for(int i = number_layers - 1;i > 0;i--){
				Adjust_Parameter(i);
			}

			int i = number_layers - 1;

			if(strstr(type_layer[i], "ce")){
				::Calculate_Loss<<<1, NUMBER_THREAD>>>(0, batch_size * number_maps[i], temporal_loss, neuron[0][i], target_output_batch);
			}
			else
			if(strstr(type_layer[i], "mse")){
				::Calculate_Loss<<<1, NUMBER_THREAD>>>(1, batch_size * number_maps[i], temporal_loss, neuron[0][i], target_output_batch);
			}
		}
	}

	for(int h = 0;h < number_parameter_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(Access_Parameter(h, i) && strstr(type_layer[i], "bn")){
				::Multiply<<<number_maps[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_maps[i], sum_mean[h][i],		(double)batch_size / number_training, mean[h][i]);
				::Multiply<<<number_maps[i] / NUMBER_THREAD + 1, NUMBER_THREAD>>>(number_maps[i], sum_variance[h][i],	(double)batch_size / (batch_size - 1) * batch_size / number_training, variance[h][i]);
			}
		}
	}

	cudaMemcpy(&loss, temporal_loss, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(target_output_batch);
	cudaFree(temporal_loss);
	delete[] index;

	return loss / number_training;
}
