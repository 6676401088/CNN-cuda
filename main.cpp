#include <stdio.h>
#include <string.h>
#include <time.h>

#include "CNN.cuh"

void Read_Data(char data[], char folder_path[], int number_train, int number_test, float **input, float **target_output){
	char path[6][255];

	if(!strcmp(data, "CIFAR-10")){
		char *filename[] = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"};

		for(int g = 0, h = 0;g < 6;g++){
			FILE *file;

			strcpy(path[g], folder_path);
			strcat(path[g], filename[g]);
			file = fopen(path[g], "rb");

			if(file){
				for(int i = 0;i < 10000;h++, i++){
					if(h == number_train && g < 4){
						g = 4;
						break;
					}
					if(h == number_train + number_test){
						g = 5;
						break;
					}

					unsigned char value;

					fread(&value, sizeof(unsigned char), 1, file);

					for(int j = 0;j < 10;j++){
						target_output[h][j] = (j == value);
					}
					for(int j = 0;j < 3 * 32 * 32;j++){
						fread(&value, sizeof(unsigned char), 1, file);
						input[h][j] = value / 255.0;
					}
				}
				fclose(file);
			}
			else{
				fprintf(stderr, "[Read_Data], %s not found.\n", path[g]);
			}
		}
	}
	else
	if(!strcmp(data, "MNIST")){
		char *filename[] = {"train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"};

		FILE *file;

		for(int g = 0;g < 4;g++){
			strcpy(path[g], folder_path);
			strcat(path[g], filename[g]);
		}
		file = fopen(path[0], "rb");

		if(file){
			for(int h = 0, value;h < 4;h++){
				fread(&value, sizeof(int), 1, file);
			}
			for(int h = 0;h < number_train;h++){
				unsigned char pixel;

				for(int j = 0;j < 28 * 28;j++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j] = pixel / 255.0;
				}
			}
			fclose(file);
		}
		else{
			fprintf(stderr, "[Read_Data], %s not found.\n", path[0]);
		}

		file = fopen(path[1], "rb");

		if(file){
			for(int h = 0, value;h < 2;h++){
				fread(&value, sizeof(int), 1, file);
			}
			for(int h = 0;h < number_train;h++){
				unsigned char label;

				fread(&label, sizeof(unsigned char), 1, file);

				for(int j = 0;j < 10;j++){
					target_output[h][j] = (j == label);
				}
			}
			fclose(file);
		}
		else{
			fprintf(stderr, "[Read_Data], %s not found.\n", path[1]);
		}

		file = fopen(path[2], "rb");

		if(file){
			for(int h = 0, value;h < 4;h++){
				fread(&value, sizeof(int), 1, file);
			}
			for(int h = number_train;h < number_train + number_test;h++){
				unsigned char pixel;

				for(int j = 0;j < 28 * 28;j++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j] = pixel / 255.0;
				}
			}
			fclose(file);
		}
		else{
			fprintf(stderr, "[Read_Data], %s not found.\n", path[2]);
		}

		file = fopen(path[3], "rb");

		if(file){
			for(int h = 0, value;h < 2;h++){
				fread(&value, sizeof(int), 1, file);
			}
			for(int h = number_train;h < number_train + number_test;h++){
				unsigned char label;

				fread(&label, sizeof(unsigned char), 1, file);

				for(int j = 0;j < 10;j++){
					target_output[h][j] = (j == label);
				}
			}
			fclose(file);
		}
		else{
			fprintf(stderr, "[Read_Data], %s not found.\n", path[3]);
		}
	}
}

int main(){
	char *type_layer[] = {"CIFAR-10", "Cbn,fs3 /sc",
							"Cbn,fs3",     "Cbn,fs3 /sc2",		"Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
							"Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
							"Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
							"Pavg", "Lce,sm"};

	int batch_size		 = 50;
	int length_map[]	 = {32,	32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16,  8,  8,  8,  8,  8,  8,  1,  1};
	int number_map[]	 = { 3, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 10};
	int number_iteration = 100;
	int number_layer	 = sizeof(type_layer) / sizeof(type_layer[0]);
	int number_train	 = 50000;
	int number_test		 = 10000;

	float epsilon		= 0.001;
	float learning_rate	= 0.002;

	float **input			= new float*[number_train + number_test];
	float **target_output	= new float*[number_train + number_test];

	Convolutional_Neural_Networks_CUDA *CNN = new Convolutional_Neural_Networks_CUDA(type_layer, number_layer, length_map, number_map);

	for(int h = 0;h < number_train + number_test;h++){
		input[h]		 = new float[number_map[0] * length_map[0] * length_map[0]];
		target_output[h] = new float[number_map[number_layer - 1]];
	}
	if(!strcmp(type_layer[0], "CIFAR-10"))	Read_Data("CIFAR-10", "", number_train, number_test, input, target_output);
	if(!strcmp(type_layer[0], "MNIST"))		Read_Data("MNIST", "", number_train, number_test, input, target_output);

	CNN->Initialize_Parameter(0, 0.2, -0.1);

	for(int g = 0, time = clock();g < number_iteration;g++){
		int number_correct[2] = {0, };

		float loss = CNN->Train(batch_size, number_train, epsilon, learning_rate, input, target_output);	

		float **output = new float*[batch_size];
		
		for(int h = 0;h < batch_size;h++){
			output[h] = new float[number_map[number_layer - 1]];
		}
		// Do not calculate the accuracy of training data to reduce computational cost.
		for(int i = number_train;i < number_train + number_test;i += batch_size){
			int test_batch_size = (i + batch_size < number_train + number_test) ? (batch_size):(number_train + number_test - i);

			CNN->Test(test_batch_size, &input[i], output);

			for(int h = 0;h < test_batch_size;h++){
				int argmax;

				float max = 0;

				for(int j = 0;j < number_map[number_layer - 1];j++){
					if(max < output[h][j]){
						argmax = j;
						max = output[h][j];
					}
				}
				number_correct[(h + i < number_train) ? (0):(1)] += (int)target_output[h + i][argmax];
			}
		}
		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", number_correct[0], number_train, number_correct[1], number_test, loss, g + 1, (float)(clock() - time) / CLOCKS_PER_SEC);
		learning_rate *= 0.955;

		for(int h = 0;h < batch_size;h++){
			delete[] output[h];
		}
		delete[] output;
	}

	for(int h = 0;h < number_train + number_test;h++){
		delete[] input[h];
		delete[] target_output[h];
	}
	delete[] input;
	delete[] target_output;
	delete CNN;

	return 0;
}
