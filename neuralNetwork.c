/*
        A basic neural network written in base C that learned the
        behavior of an XOR logic function

        2023 - Kyle Levy

        Based on: https://www.youtube.com/watch?v=LA4I3cWkp1E&ab_channel=NicolaiNielsen
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double init_weights() {
    /* Returns a random value normalized between 0 and 1 */
    /* as a initial weight for a neuron connection       */
    return ((double)rand()) / ((double)RAND_MAX); 
}

double sigmoid(double x){
    /* Sigmoid function to squash values into range of 0-1 */
    return 1 / (1 + exp(-x));
}

double dSigmoid(double x){
    /* The derrivative of the sigmoid function at x */
    return x * (1-x);
}

void shuffle(int *array, size_t n){
    /* This function shuffles an array */
    if (n > 1){
        size_t i;
        for (i = 0; i < n - 1; ++i){
            size_t j = i + rand() / (RAND_MAX / (n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void) {

    // Declaring the learning rate for the model
    const double lr = 0.1f;

    // Initializing arrays to store the nodes of the model
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    // Initializing an array to store the bias of each node
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    // Initializing an array to store the weights between each node
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    // Iniitalizing the training data
    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}};

    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

    // Initializing the weights
    for(int i=0; i < numInputs; i++){
        for(int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = init_weights();
        }
    }

    for(int i=0; i < numHiddenNodes; i++){
        for(int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = init_weights();
        }
    }

    for(int i = 0; i < numOutputs; i++){
        outputLayerBias[i] = init_weights();
    }

    // Declaring the training perameters
    int trainingSetOrder[] = {0, 1, 2, 3};

    int numberOfEpochs = 10000;

    /* Training the neural network */ 
    for(int epochs=0; epochs < numberOfEpochs; epochs++){
        printf("Epoch %d training...\n", epochs);

        shuffle(trainingSetOrder, numTrainingSets);

        for(int x=0; x < numTrainingSets; x++){
            int i = trainingSetOrder[x];

            /* Forward Pass */

            /* Compute hidden layer activation */
            for(int j=0; j < numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k=0; k < numInputs; k++){
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            /* Compute output layer activation */
            for(int j=0; j < numOutputs; j++){
                double activation = outputLayerBias[j];

                for(int k=0; k < numHiddenNodes; k++){
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g %g  Predicted Output: %g  Expected Output: %g \n", 
                    training_inputs[i][0], training_inputs[i][1],
                    outputLayer[0], training_outputs[i][0]);

            /* Back Propogation */
            
            /* Compute change in output weights */
            double deltaOutput[numOutputs];

            for(int j=0; j < numOutputs; j++){
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            /* Compute change in hidden weights */
            double deltaHidden[numHiddenNodes];
            for(int j=0; j < numHiddenNodes; j++){
                double error = 0.0f;
                for(int k=0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            /* Apply changes in output weights */
            for(int j=0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k=0; k < numHiddenNodes; k++){
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            /* Apply changes in hidden weights*/
            for(int j=0; j < numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k=0; k < numInputs; k++){
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
    
    /* Print final weights and biases */
    fputs("Final Hidden Weights\n[ ", stdout);
    for(int j=0; j < numHiddenNodes; j++){
        fputs("[ ", stdout);
        for(int k=0; k < numInputs; k++){
            printf("%f ", hiddenWeights[k][j]);
        }
        fputs(" ]", stdout);
    }
    fputs(" ]\n", stdout);

    fputs("Final Hidden Biases\n[ ", stdout);
    for(int j=0; j < numHiddenNodes; j++){
        fputs("[ ", stdout);
        printf("%f ", hiddenLayerBias[j]);
        fputs(" ]", stdout);
    }
    fputs(" ]\n", stdout);

    fputs("Final Output Weights\n[ ", stdout);
    for(int j=0; j < numOutputs; j++){
        fputs("[ ", stdout);
        for(int k=0; k < numHiddenNodes; k++){
            printf("%f ", outputWeights[k][j]);
        }
        fputs(" ]", stdout);
    }
    fputs(" ]\n", stdout);

    fputs("Final Output Biases\n[ ", stdout);
    for(int j=0; j < numOutputs; j++){
        fputs("[ ", stdout);
        printf("%f ", outputLayerBias[j]);
        fputs(" ]", stdout);
    }
    fputs(" ]\n", stdout);

    return 0;

}