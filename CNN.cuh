#include "cuda_runtime.h"

__global__ void Activate(int option, int number_neurons, float neuron[]);
__global__ void Add(int number_memory, float A[], float B[], float C[]);
__global__ void Adjust_Parameter(int batch_size, int lower_layer_index, int layer_index, int kernel_width, int kernel_height, int stride_width, int stride_height, int map_width[], int map_height[], int number_maps[], float derivative[], float lower_neuron[], float weight[]);
__global__ void Backpropagate(int option, int batch_size, int layer_index, int upper_layer_index, int upper_kernel_width, int upper_kernel_height, int upper_stride_width, int upper_stride_height, int map_width[], int map_height[], int number_maps[], float derivative[], float upper_derivative[], float upper_weight[]);
__global__ void Batch_Normalization_Activate(int option, int batch_size, int map_width, int map_height, int number_maps, float epsilon, float gamma[], float beta[], float mean[], float variance[], float sum_mean[], float sum_variance[], float neuron[], float neuron_batch_0[], float neuron_batch_1[]);
__global__ void Batch_Normalization_Adjust_Parameter(int batch_size, int map_width, int map_height, int number_maps, float gamma[], float beta[], float derivative_batch_1[], float neuron_batch_0[]);
__global__ void Batch_Normalization_Differentiate(int batch_size, int map_width, int map_height, int number_maps, float epsilon, float gamma[], float beta[], float mean[], float variance[], float derivative[], float derivative_batch_0[], float derivative_batch_1[], float neuron_batch_1[]);
__global__ void Calculate_Loss(int option, int number_neurons, float *error, float neuron[], float target_output[]);
__global__ void Differentiate(int option, int number_memory, float learning_rate, float derivative[], float neuron[], float target_output[]);
__global__ void Dropout(int option, int batch_size, int number_maps, int map_width, int map_height, int seed, float rate, float neuron[]);
__global__ void Feedforward(int option, int batch_size, int lower_layer_index, int layer_index, int kernel_width, int kernel_height, int stride_width, int stride_height, int number_maps[], int map_width[], int map_height[], float lower_neuron[], float neuron[], float weight[]);
__global__ void Multiply(int number_memory, float A[], float value, float B[]);
__global__ void Multiply(int number_memory, float A[], float B[], float C[]);
__global__ void Randomize(int number_memory, int seed, float scale, float shift, float A[]);
__global__ void Set(int number_memory, float value, float A[]);
__global__ void Softmax(int batch_size, int map_width, int map_height, int number_maps, float neuron[]);

class Convolutional_Neural_Networks_CUDA{
private:
	char **type_layer;
	
	int batch_size;
	int number_layers;
	int number_memory_types;
	int number_parameter_types;

	int *kernel_width;
	int *kernel_height;
	int *map_width;
	int *map_width_factor;
	int *map_height;
	int *map_height_factor;
	int *number_maps;
	int *number_maps_factor;
	int *stride_width;
	int *stride_height;

	float ***derivative;
	float ***neuron;
	float ***weight;

	// Variables for Batch Normalization
	float epsilon;

	float ***gamma;
	float ***beta;
	float ***mean;
	float ***variance;
	float ***sum_mean;
	float ***sum_variance;
	// *********************************

	// Variables for Residual Learning
	int *shortcut_kernel_width;
	int *shortcut_kernel_height;
	int *shortcut_stride_width;
	int *shortcut_stride_height;
	// *******************************
	
	void Activate(char option[], int layer_index);
	void Adjust_Parameter(int layer_index);
	void Backpropagate(int layer_index);
	void Differentiate(int layer_index, float learning_rate, float target_output[]);
	void Feedforward(int layer_index);

	void Batch_Normalization_Activate(char option[], char type[], int layer_index);
	void Batch_Normalization_Adjust_Parameter(char type[], int layer_index);
	void Batch_Normalization_Differentiate(char type[], int layer_index);

	void Resize_Memory(int batch_size);

	bool Access_Memory(int type_index, int layer_index);
	bool Access_Parameter(int type_index, int layer_index);
public:
	Convolutional_Neural_Networks_CUDA(char **type_layer, int number_layers, int map_width[], int map_height[], int number_maps[]);
	~Convolutional_Neural_Networks_CUDA();

	void Initialize_Parameter(int seed, double scale, double shift);
	void Load_Parameter(char path[]);
	void Save_Parameter(char path[]);
	void Test(float input[], float output[]);
	void Test(int batch_size, float **input, float **output);

	float Train(int batch_size, int number_training, float epsilon, float learning_rate, float **input, float **target_output);
};
