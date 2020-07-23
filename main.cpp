#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include "Aria.h"

using namespace std;

//training data inputs and outputs
struct trainingData {
	vector<double> inputs;
	vector<double> outputs;
};

//neuron, its outputs, the delta of its weights, its gradient, and its activation function
class Neuron {       
public:             
	double input;
	vector<double> outputs;
	vector<double> deltaWeights;
	double gradient;
	void activationFunc(double lambda) {  
		input = 1/(1+exp(-1*lambda*input));
	}
};

//neural network class
class NeuralNetwork {       
public:
	//used to store the neurons and their weights in the network
	vector<vector<Neuron>> neuronVector;

	//used to store the training, testing, and validating data
	vector<trainingData> trainingVector;
	vector<trainingData> testingVector;
	vector<trainingData> validatingVector;

	//used to assign the store of the neurons
	vector<double> NeuronOutputs;

	//used to help calculate gradient and momentum, to save previous weights, and to store in savedWeights.csv
	vector<double> deltaWeightsRecord;
	vector<double> weightsRecord;

	//tuning parameters
	double eta;
	double lambda;
	double alpha;
	double epochs;

	//Used to get the min and max of the read data to normalize
	double maxLeftDist = 0;
	double minLeftDist = 9999;
	double maxFrontDist = 0;
	double minFrontDist = 9999;
	double maxLeftMotor = 0;
	double minLeftMotor = 9999;
	double maxRightMotor = 0;
	double minRightMotor = 9999;

	//stopping criteria parameters, if validation error increases x amount of times, stop there
	double prevVal = 9999;
	int valIncrCountOrig;
	int valIncrCount;

	//used to store the RMSE
	vector<double> trainRMSE = {0,0};
	vector<double> validateRMSE = {0,0};
	vector<double> testRMSE = {0,0};

	//used to show the final testing value for tuning
	double finalTest = 0;

	//used to calculate the robot speed, using predict and normalization
	pair<double,double> robotOutput(double input1, double input2)
	{

		input1 = (input1 - minLeftDist) / (maxLeftDist - minLeftDist);
		input2 = (input2 - minFrontDist) / (maxFrontDist - minFrontDist);

		vector<double> result = predict({ input1, input2 });
		return{ result[0] * (maxLeftMotor - minLeftMotor) + minLeftMotor, result[1] * (maxRightMotor - minRightMotor) + minRightMotor };
	}
	
	//used to try different parameters to find the best combination
	void tune(){
		eta = 0;
		alpha = 0;
		lambda = 0;

		vector<vector<Neuron>> backupVector = neuronVector;

		double finalEta = 0;
		double finalAlpha = 0;
		double finalLambda = 0;

		double biggestError = 9999;

		ofstream tuneFile("tuning.csv");

		tuneFile << "eta" << ',' << "alpha" << ',' << "lambda" << ',' << "testRMSE" << endl;

		for(int i = 0; i < 10; i++){
			eta += 0.1;
			alpha = 0;
			lambda = 0;
			for(int j = 0; j < 10; j++){
				alpha += 0.1;
				lambda = 0;
				for(int k = 0; k < 10; k++){
					lambda += 0.1;

					NeuronOutputs = {};
					deltaWeightsRecord= {};
					weightsRecord = {};

					trainRMSE = {0,0};
					validateRMSE = {0,0};
					testRMSE = {0,0};

					prevVal = 9999;
					valIncrCount = valIncrCountOrig;

					neuronVector = backupVector;

					train();

					cout << "eta = " << eta << ", alpha = " << alpha << ", lambda = " << lambda << ", testRMSE = " << finalTest << endl;

					tuneFile << eta << ',' << alpha << ',' << lambda << ',' << finalTest << endl;

					if(finalTest < biggestError){
						biggestError = finalTest;
						finalAlpha = alpha;
						finalEta = eta;
						finalLambda = lambda;
					}
				}
			}
		}

		cout << biggestError << " " << finalEta << " " << finalAlpha << " " << finalLambda << endl;
	}

	//used to show the testRMSE
	void test(){
		for(int i = 0; i < testingVector.size(); i++){
			vector<double> result = predict(testingVector[i].inputs);

			double leftError = testingVector[i].outputs[0] - result[0];
			testRMSE[0] += leftError*leftError;
			
			double rightError = testingVector[i].outputs[1] - result[1];
			testRMSE[1] += rightError*rightError;	

		}
		double LeftRMSE = sqrt(testRMSE[0] / testingVector.size());
		double RightRMSE = sqrt(testRMSE[1] / testingVector.size());
		double finalRMSE = (LeftRMSE + RightRMSE) / 2;
		cout << "Test RMSE = " << finalRMSE << endl;

		finalTest = finalRMSE;
	}

	//used to show the validateRMSE
	void validate(){
		for(int i = 0; i < validatingVector.size(); i++){
			vector<double> result = predict(validatingVector[i].inputs);

			double leftError = validatingVector[i].outputs[0] - result[0];
			validateRMSE[0] += leftError*leftError;
			
			double rightError = validatingVector[i].outputs[1] - result[1];
			validateRMSE[1] += rightError*rightError;
		}

		double LeftRMSE = sqrt(validateRMSE[0] / validatingVector.size());
		double RightRMSE = sqrt(validateRMSE[1] / validatingVector.size());
		double finalRMSE = (LeftRMSE + RightRMSE) / 2;
		cout << "Validate RMSE = " << finalRMSE << endl;
		
		if(finalRMSE > prevVal){
			valIncrCount--;
		}
		else{
			valIncrCount = valIncrCountOrig;
		}
		prevVal = finalRMSE;
	}

	//used to save the weights in a csv, to avoid training on the robot
	//also stores the max and min values to aid on normalization
	void saveWeights(){
		ofstream f;
		f.open( "savedWeights.csv", std::ios::out);
		for(int i = 0; i < weightsRecord.size(); i++){
			f << weightsRecord[i] << ",";
		}
		f << maxLeftDist << "," << minLeftDist << "," << maxFrontDist << "," << minFrontDist << "m,";
		f << maxLeftMotor << "," << minLeftMotor << "," << maxRightMotor << "," << minRightMotor;
		f.close();
	}

	//used to load the weights in a csv, to avoid training on the robot
	void loadWeights(double numWeights){
		ifstream file("savedWeights.csv");
		string line;
		string reading;

		getline(file, line);
		stringstream lineStream(line);
		for(int i = 0; i < numWeights; i++){
			getline( lineStream, reading, ',' );
			weightsRecord.push_back(atof(reading.c_str()));
		}

		getline( lineStream, reading, ',' );
		maxLeftDist = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		minLeftDist = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		maxFrontDist = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		minFrontDist = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		maxLeftMotor = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		minLeftMotor = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		maxRightMotor = atof(reading.c_str());
		getline( lineStream, reading, ',' );
		minRightMotor = atof(reading.c_str());

		//workStart is used to not accidentally load non existing weights to bias neurons
		int deltaCounter = 0;
		for(int column = neuronVector.size() - 1; column > 0; column--){
			int workStart = 0;
			if(column != neuronVector.size() - 1)
			{
				workStart++;
			}

			for(int row = workStart; row < neuronVector[column].size(); row++){
				for(int pastNeuron = 0; pastNeuron < neuronVector[column - 1].size(); pastNeuron++){
					double weightSource = row;
					if(column != neuronVector.size() - 1)
					{
						weightSource--;
					}

					neuronVector[column - 1][pastNeuron].outputs[weightSource] = weightsRecord[deltaCounter];
					deltaCounter++;
				}
			}
		}
		
	}

	//load the training data from the csv. also store the min and max to aid in normalization
	void loadTrainingData(string fileName){
		vector<trainingData> readVector;

		ifstream file(fileName);
		string line;
		string reading;
		double leftDist;
		double frontDist;
		double leftMotor;
		double rightMotor;
		getline(file, line);

		while(file){
			getline(file, line);
			stringstream lineStream(line);
			getline( lineStream, reading, ',' );
			leftDist = atof(reading.c_str());
			getline( lineStream, reading, ',' );
			frontDist = atof(reading.c_str());
			getline( lineStream, reading, ',' );
			leftMotor = atof(reading.c_str());
			getline( lineStream, reading, ',' );
			rightMotor = atof(reading.c_str());
			
			
			if(leftDist != 0 && frontDist != 0 && leftMotor != 0 && rightMotor != 0){
				readVector.push_back({{leftDist, frontDist}, {leftMotor, rightMotor}});
			}

		}

		for (int i = 0; i < readVector.size(); i++) {

			leftDist = readVector[i].inputs[0];
			frontDist = readVector[i].inputs[1];
			leftMotor = readVector[i].outputs[0];
			rightMotor = readVector[i].outputs[1];

			if (leftDist > maxLeftDist) {
				maxLeftDist = leftDist;
			}
			if (leftDist < minLeftDist) {
				minLeftDist = leftDist;
			}
			if (frontDist > maxFrontDist) {
				maxFrontDist = frontDist;
			}
			if (frontDist < minFrontDist) {
				minFrontDist = frontDist;
			}

			if (leftMotor > maxLeftMotor) {
				maxLeftMotor = leftMotor;
			}
			if (leftMotor < minLeftMotor) {
				minLeftMotor = leftMotor;
			}
			if (rightMotor > maxRightMotor) {
				maxRightMotor = rightMotor;
			}
			if (rightMotor < minRightMotor) {
				minRightMotor = rightMotor;
			}

		}

		//use the min and max values to normalize
		scaleWeights(readVector);

		//shuffle the data to prevent bias
		random_shuffle(readVector.begin(), readVector.end());

		//70% training, 15% validation, 15% testing
		double trainSize = readVector.size()*0.7;
		double validateSize = readVector.size()*0.15;
		
		vector<trainingData> tempTrainVector(readVector.begin(), readVector.begin() + trainSize);
		trainingVector = tempTrainVector;
		vector<trainingData> tempValidateVector(readVector.begin() + trainSize, readVector.begin() + trainSize + validateSize);
		validatingVector = tempValidateVector;
		vector<trainingData> tempTestVector(readVector.begin() + trainSize + validateSize, readVector.end());
		testingVector = tempTestVector;
	
	}

	//predict (estimate) the motor speeds based on a set of points
	vector<double> predict(vector<double> data){

		for(int x = 0; x < data.size(); x++){
			neuronVector[0][x+1].input = data[x];
		}
		FeedForward();
		
		return{ neuronVector[neuronVector.size() - 1][0].input, neuronVector[neuronVector.size() - 1][1].input};
		
	}

	//train the neural network
	void train() {

		trainRMSE = {0,0};
		validateRMSE = {0,0};
		testRMSE = {0,0};

		for(int i = 0; i < epochs; i++){
			//shuffle before beginning each epoch
			random_shuffle( trainingVector.begin(), trainingVector.end());

			trainRMSE[0] = 0;
			trainRMSE[1] = 0;

			validateRMSE[0] = 0;
			validateRMSE[1] = 0;
			
			for(int j = 0; j < trainingVector.size(); j++){
				NeuronOutputs = trainingVector[j].outputs;
				for(int x = 0; x < trainingVector[j].inputs.size(); x++){
					neuronVector[0][x+1].input = trainingVector[j].inputs[x];
				}
				FeedForward();
				BackPropagation();
			}
			double finalRMSE = (sqrt(trainRMSE[0] / trainingVector.size()) + sqrt(trainRMSE[1] / trainingVector.size())) / 2;
			cout << "Train RMSE = " << finalRMSE << endl;
			validate();

			//if the stopping criteria is met (x amount of error increasing in a row), stop there
			if(valIncrCount == 0){
				break;
			}
		}

		test();

	}

	//scale (normalize) the weights using the min and max values
	void scaleWeights(vector<trainingData> &readVector) {
		for (int i = 0; i < readVector.size(); i++) {
			readVector[i].inputs[0] = (readVector[i].inputs[0] - minLeftDist) / (maxLeftDist - minLeftDist);
			readVector[i].inputs[1] = (readVector[i].inputs[1] - minFrontDist) / (maxFrontDist - minFrontDist);
			readVector[i].outputs[0] = (readVector[i].outputs[0] - minLeftMotor) / (maxLeftMotor - minLeftMotor);
			readVector[i].outputs[1] = (readVector[i].outputs[1] - minRightMotor) / (maxRightMotor - minRightMotor);
		}

	}

	//do feed forward
	void FeedForward() {  
		for(int column = 0; column < neuronVector.size() - 1; column++){

			//like workStart, used to avoid colliding with the (non existing) bias on the output layer
			int workSize = neuronVector[column + 1].size();
			if(column != neuronVector.size() - 2)
			{
				workSize--;
			}
			
			for(int i = 0; i < workSize; i++)
				{
					double newInput = 0;

					for(int j = 0; j < neuronVector[column].size(); j++)
					{
						newInput += (neuronVector[column][j].input * neuronVector[column][j].outputs[i]);
					}

					int newI = i;
					if (column != neuronVector.size() - 2) {
						newI = i + 1;
					}

					neuronVector[column+1][newI].input = newInput;
					neuronVector[column+1][newI].activationFunc(lambda);

				}	
		}
	}

	//do back propagation
	void BackPropagation() {
		vector<double> deltaWeightVector;

		for(int column = neuronVector.size() - 1; column > 0; column--){
			int workStart = 0;
			if(column != neuronVector.size() - 1)
			{
				workStart++;
			}
			
			for(int row = workStart; row < neuronVector[column].size(); row++){
				double gradient = 0;
				if(column == neuronVector.size() - 1){
					double error = NeuronOutputs[row] - neuronVector[column][row].input;
					trainRMSE[row] += error*error;
					
					gradient = lambda * neuronVector[column][row].input * (1-neuronVector[column][row].input) * error;
					neuronVector[column][row].gradient = gradient;
				}
				else{
					double gradientsSum = 0;
					
					for(int x = 0; x < neuronVector[column][row].outputs.size(); x++){
						gradientsSum += neuronVector[column][row].outputs[x] * neuronVector[column+1][x].gradient;
					}
					gradient = lambda * neuronVector[column][row].input * (1-neuronVector[column][row].input) * ( gradientsSum );
					neuronVector[column][row].gradient = gradient;
				}

				//used for momentum
				for(int pastNeuron = 0; pastNeuron < neuronVector[column - 1].size(); pastNeuron++){
					double deltaWeight = eta * gradient * neuronVector[column - 1][pastNeuron].input;
					deltaWeightVector.push_back(deltaWeight);
				}
			}
		}

		//update weights
		int deltaCounter = 0;
		for(int column = neuronVector.size() - 1; column > 0; column--){
			int workStart = 0;
			if(column != neuronVector.size() - 1)
			{
				workStart++;
			}

			for(int row = workStart; row < neuronVector[column].size(); row++){
				for(int pastNeuron = 0; pastNeuron < neuronVector[column - 1].size(); pastNeuron++){
					double weightSource = row;
					if(column != neuronVector.size() - 1)
					{
						weightSource--;
					}

					//add momentum if it exists
					if(deltaWeightsRecord.size() == 0){
						neuronVector[column - 1][pastNeuron].outputs[weightSource] += deltaWeightVector[deltaCounter];
						weightsRecord.push_back(neuronVector[column - 1][pastNeuron].outputs[weightSource]);
					}
					else{
						neuronVector[column - 1][pastNeuron].outputs[weightSource] += deltaWeightVector[deltaCounter] + (alpha * deltaWeightsRecord[deltaCounter]);
						weightsRecord[deltaCounter] = (neuronVector[column - 1][pastNeuron].outputs[weightSource]);
					}
					deltaCounter++;
				}
			}
		}

		//store the new delta weights
		deltaWeightsRecord = deltaWeightVector;

	}


};

int main(int argc, char **argv)
{
	
	NeuralNetwork network;

	network.epochs = 150;
	network.loadTrainingData("nntrainingdata.csv");

	//6 hidden layer neurons
	network.neuronVector = { { { 1,{ 0.1, 0.7, 0.3, 0.2, 0.4, 0.3 } } ,{ -999999,{ 0.6, 0.35, 0.1, 0.6, 0.2, 0.6 } } , { -999999,{ 0.3, 0.8, 0.7, 0.2, 0.3, 0.5 } } }, { { 1,{ 0.3, 0.1 } } ,{ -999999,{ 0.4, 0.5 } } , { -999999,{ 0.7, 0.3 } },{ -999999,{ 0.1, 0.4 } } ,{ -999999,{ 0.2, 0.4 } }, { -999999,{ 0.5, 0.3 } }, { -999999,{ 0.5, 0.7 } }}, { { -999999,{ -999999 } }, { -999999,{ -999999 } } } };

	//4 hidden layer neurons
	//network.neuronVector = { { { 1,{ 0.1, 0.7, 0.3, 0.2 } } ,{ -999999,{ 0.6, 0.35, 0.1, 0.6 } } , { -999999,{ 0.3, 0.8, 0.7, 0.2 } } }, { { 1,{ 0.3, 0.1 } } ,{ -999999,{ 0.4, 0.5 } } , { -999999,{ 0.7, 0.3 } },{ -999999,{ 0.1, 0.4 } } ,{ -999999,{ 0.2, 0.4 } } }, { { -999999,{ -999999 } }, { -999999,{ -999999 } } } };

	//3 hidden layer neurons
	//network.neuronVector = { { { 1,{ 0.1, 0.7, 0.3 } } ,{ -999999,{ 0.6, 0.35, 0.1 } } , { -999999,{ 0.3, 0.8, 0.7 } } }, { { 1,{ -0.3, 0.1 } } ,{ -999999,{ 0.4, 0.5 } } , { -999999,{ 0.7, 0.3 } },{ -999999,{ 0.1, 0.4 } } }, { { -999999,{ -999999 } }, { -999999,{ -999999 } } } };

	//2 hidden layer neurons
	//network.neuronVector = { { { 1,{ 0.1, 0.7 } } ,{ -999999,{ 0.6, 0.35 } } , { -999999,{ 0.3, 0.8 } } }, { { 1,{ -0.3, 0.1 } } ,{ -999999,{ 0.4, 0.5 } } , { -999999,{ 0.7, 0.3 } } }, { { -999999,{ -999999 } }, { -999999,{ -999999 } } } };
	
	//tuning parameters
	network.eta = 0.9;
	network.alpha = 1;
	network.lambda = 1;
	network.valIncrCountOrig = 6;

	//to train the neural network and get weights, save them
	//network.train();
	//network.saveWeights();

	//network.tune();

	//to load previously saved weights
	network.loadWeights(32);

	//manually make a prediction for debugging
	//pair<double,double> result = network.robotOutput(343.778, 2693.63);
	//cout << result.first << " " << result.second << endl;
	
	// create instances
	Aria::init();
	ArRobot robot;

	// parse command line arguments
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();

	ArRobotConnector robotConnector(&argParser, &robot);

	if (robotConnector.connectRobot())
	cout << "Robot connected!" << endl;
	robot.runAsync(false);

	robot.lock();
	robot.enableMotors();
	robot.unlock();

	ArSensorReading *sonarSensor[8];

	//**ROBOT SETUP & CONNECTION**

	int sonarRange[8];

	ArUtil::sleep(8000);

	while (true)
	{
	//Get Readings
	//Get Readings
		for (int i = 0; i < 8; i++) {
			sonarSensor[i] = robot.getSonarReading(i);
			int distance = sonarSensor[i]->getRange();
			sonarRange[i] = distance;
		}

		int min0 = INT_MAX;
		int min1 = INT_MAX;
		int min3 = INT_MAX;
		int min4 = INT_MAX;
		for (int i = 0; i < 10; i++)
		{
			int reading0 = robot.getSonarReading(0)->getRange();
			if (reading0 < min0)
			{

				min0 = reading0;
			}

			int reading1 = robot.getSonarReading(1)->getRange();
			if (reading1 < min1)
			{

				min1 = reading1;
			}

			int reading3 = robot.getSonarReading(3)->getRange();
			if (reading3 < min3)
			{

				min3 = reading3;
			}

			int reading4 = robot.getSonarReading(4)->getRange();
			if (reading4 < min4)
			{

				min4 = reading4;
			}
		}

		sonarRange[0] = min0;
		sonarRange[1] = min1;
		sonarRange[3] = min3;
		sonarRange[4] = min4;


	double frontSonar = min((sonarRange[3]), sonarRange[4]);
	double leftSonar = min((sonarRange[0]), sonarRange[1]);

	pair<double,double> robotResult = network.robotOutput( leftSonar, frontSonar);

	robot.setVel2(robotResult.first, robotResult.second);

	ArUtil::sleep(100);
	}



	// stop the robot
	robot.lock();
	robot.stop();
	robot.unlock();

	// terminate all threads and exit
	Aria::exit();



	
	return 0;
}