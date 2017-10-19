#include <stdio.h>
#include <string.h>
#include <time.h>

#include "RNN.cuh"

void Read_MNIST(char training_set_images[], char training_set_labels[], char test_set_images[], char test_set_labels[], int time_step, int number_training, int number_test, float ***input, float ***target_output){
	FILE *file;

	if(file = fopen(training_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char pixel;

			for(int j = 0;j < time_step;j++){
				for(int k = 0;k < 784 / time_step;k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", training_set_images);
	}

	if(file = fopen(training_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int t = 0;t < time_step;t++){
				for(int j = 0;j < 10;j++){
					target_output[h][t][j] = (j == label);
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", training_set_labels);
	}

	if(file = fopen(test_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char pixel;

			for(int j = 0;j < time_step;j++){
				for(int k = 0;k < 784 / time_step;k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", test_set_images);
	}

	if(file = fopen(test_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int t = 0;t < time_step;t++){
				for(int j = 0;j < 10;j++){
					target_output[h][t][j] = (j == label);
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", test_set_labels);
	}
}

int main(){
	char *type_layer[] = {"MNIST", "Clstm", "Lce,sm"};

	int batch_size			= 60;
	int time_step			= 28;
	int map_width[]			= {1, 1, 1};
	int map_height[]		= {1, 1, 1};
	int number_maps[]		= {784 / time_step, 100, 10};
	int number_iterations	= 100;
	int number_layers		= sizeof(type_layer) / sizeof(type_layer[0]);
	int number_training		= 60000;
	int number_test			= 10000;
	
	int *length_data = new int[number_training + number_test];
	int *output_mask = new int[time_step];

	float epsilon				= 0.001;
	float gradient_threshold	= 10;
	float learning_rate			= 0.005; // 0.001 for vanilla RNN
	float decay_rate			= 0.977;

	float ***input			= new float**[number_training + number_test];
	float ***target_output	= new float**[number_training + number_test];

	Recurrent_Neural_Networks_CUDA RNN = Recurrent_Neural_Networks_CUDA(type_layer, number_layers, map_width, map_height, number_maps);

	for(int h = 0;h < number_training + number_test;h++){
		length_data[h] = time_step;

		input[h]		 = new float*[length_data[h]];
		target_output[h] = new float*[length_data[h]];

		for(int t = 0;t < length_data[h];t++){
			input[h][t]			= new float[number_maps[0] * map_height[0] * map_width[0]];
			target_output[h][t] = new float[number_maps[number_layers - 1]];
		}
	}
	for(int t = 0;t < time_step;t++){
		output_mask[t] = (t == time_step - 1);
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", time_step, number_training, number_test, input, target_output);	

	RNN.Initialize_Parameter(0, 0.2, -0.1);

	for(int g = 0, time = clock();g < number_iterations;g++){
		int number_correct[2] = {0, };

		float loss = RNN.Train(batch_size, number_training, time_step, length_data, output_mask, epsilon, gradient_threshold, learning_rate, input, target_output);

		float **_input = new float*[batch_size];
		float **output = new float*[batch_size];

		for(int h = 0;h < batch_size;h++){
			_input[h] = new float[number_maps[0] * map_height[0] * map_width[0]];
			output[h] = new float[number_maps[number_layers - 1]];
		}
		
		for(int i = 0;i < number_training + number_test;i += batch_size){
			int test_batch_size = (i + batch_size < number_training + number_test) ? (batch_size):(number_training + number_test - i);

			for(int t = 0;t < time_step;t++){
				for(int h = 0;h < test_batch_size;h++){
					memcpy(_input[h], input[i + h][t], sizeof(float) * number_maps[0] * map_height[0] * map_width[0]);
				}
				RNN.Test(t == 0, test_batch_size, _input, output);

				if(t == time_step - 1){
					for(int h = 0;h < test_batch_size;h++){
						int argmax;

						float max = 0;

						for(int j = 0;j < number_maps[number_layers - 1];j++){
							if(max < output[h][j]){
								argmax = j;
								max = output[h][j];
							}
						}
						number_correct[(h + i < number_training) ? (0):(1)] += (int)target_output[h + i][t][argmax];
					}
				}
			}
		}
		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, loss, g + 1, (float)(clock() - time) / CLOCKS_PER_SEC);
		learning_rate *= decay_rate;

		for(int h = 0;h < batch_size;h++){
			delete[] _input[h];
			delete[] output[h];
		}
		delete[] _input;
		delete[] output;
	}

	for(int h = 0;h < number_training + number_test;h++){
		for(int t = 0;t < length_data[h];t++){
			delete[] input[h][t];
			delete[] target_output[h][t];
		}		
		delete[] input[h];
		delete[] target_output[h];
	}
	delete[] input;
	delete[] target_output;
	delete[] length_data;
	delete[] output_mask;

	return 0;
}
