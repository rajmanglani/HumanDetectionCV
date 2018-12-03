/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package humandetectioncv;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author rmanglani
 */
public class NeuralNetwork {
    
    private Map<Double[], Double> trainData;
    private Double[] input;
    private Double output;
    private double learningRate;
    private int hiddenLayerSize; //250
    private double[][] inputHiddenWeights;
    private double[] hiddenOutputWeights;
    
    private double[] hiddenLayerInputs;
    private double[] hiddenLayerOutputs;
    private double outputLayerInput = 0.0;
    private double outputLayerOutput = 0.0;
    private List<Double> squaredErrorsEpoch;

    public NeuralNetwork(Map<Double[],Double> input, double learningRate, int hiddenLayerSize, int inputLayerSize) {
        this.trainData = input;
        this.learningRate = learningRate;
        this.hiddenLayerSize = hiddenLayerSize;
        inputHiddenWeights = new double[inputLayerSize][hiddenLayerSize];   // weight from every input to the hidden layer neuron 
        for(int i=0; i<inputHiddenWeights.length; i++){
            for(int j=0; j<inputHiddenWeights[i].length; j++){
                inputHiddenWeights[i][j] =  0.5 - Math.random();
            }
        }
        hiddenOutputWeights = new double[hiddenLayerSize];
        for(int i=0; i<hiddenOutputWeights.length; i++){
            hiddenOutputWeights[i] = 0.5 - Math.random();
        }
        hiddenLayerInputs = new double[hiddenLayerSize];
        hiddenLayerOutputs = new double[hiddenLayerSize]; 
        squaredErrorsEpoch = new ArrayList<>();
    }

    public void trainNetwork() throws IOException{
        for(int epoch = 0; epoch < 100; epoch++){
            System.out.println("Epoch .. " + epoch);
            //shuffle the inputs to feed in 
            List<Double[]> keys = new ArrayList(trainData.keySet());
            Collections.shuffle(keys);
            for (Double[] trainIn : keys) {
                // Access keys/values in a random order
                input = trainIn;
                output = trainData.get(trainIn);
                forward();
                backprop();
                reset();
            }
            
            System.out.println("Mean Error " + mse());
            squaredErrorsEpoch.clear();
        }
        FileOutputStream fileOut = new FileOutputStream("weights.txt");
        ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
        objectOut.writeObject(inputHiddenWeights);
        objectOut.writeObject(hiddenOutputWeights);
        objectOut.close();
        System.out.println("The Object  was succesfully written to a file");

        
    }
    
    private void reset(){
        hiddenLayerInputs = new double[hiddenLayerSize];
        hiddenLayerOutputs = new double[hiddenLayerSize]; 
        outputLayerInput = 0.0;
        outputLayerOutput = 0.0;
    }
    private double mse(){
        double sum = 0.0;
        for(Double d : squaredErrorsEpoch){
            sum += d;
        }
        return sum/squaredErrorsEpoch.size();
    }
    
    private void forward(){
        forwardPropInputToHidden();
        hiddenReLUActivation();
        forwardPropHiddenToOutput();
        outputSigmoidActivation();
    }
    
    private void backprop(){
        double error = meanSquaredError(output);
        updateWeights(outputLayerOutput - output);
    }
    
    private void forwardPropInputToHidden(){
        for(int i =0; i<hiddenLayerSize; i++){
            //calculate input times weight for each neuron 
            double sum = 0.0;
            for(int j =0; j<input.length; j++){
                sum += (input[j]*inputHiddenWeights[j][i]);
            }
            hiddenLayerInputs[i] = sum;
            //System.out.println(i + " " + hiddenLayerInputs[i]);
        }
    }
    
    private void hiddenReLUActivation(){
        //apply relu activation to all inputs f(x) = max(0,x)
        for(int i =0; i<hiddenLayerInputs.length; i++){
            if(hiddenLayerInputs[i] > 0){
                hiddenLayerOutputs[i] = hiddenLayerInputs[i];
            }else{
                hiddenLayerOutputs[i] = 0;
            }
        }
    }
    
    private void forwardPropHiddenToOutput(){
        for(int i =0; i<hiddenLayerOutputs.length; i++){
            outputLayerInput += (hiddenLayerOutputs[i] * hiddenOutputWeights[i]);
        }
        //System.out.println(outputLayerInput);
    }
    
    private void outputSigmoidActivation(){
        //System.out.println("Output layer input " + outputLayerInput);
        outputLayerOutput = 1.0 / (1 + Math.exp(-1.0 * outputLayerInput));
        //System.out.println("Output layer out " + outputLayerOutput);
    }
    
    private double meanSquaredError(double out){
//        System.out.println("Prediction " + outputLayerOutput);
//        System.out.println("Actual " + out);
        double err = 0.5*(Math.pow(out - outputLayerOutput, 2));
        squaredErrorsEpoch.add(err);
        return err;
    }
    
    private void updateWeights(double error){
        //for hidden to output layer 
        //calculate delta - error times derivative of sigmoid 
        double deltaOutput = error * (outputLayerOutput * (1-outputLayerOutput));
        //System.out.println("Delta output " + deltaOutput);
        //update weights 
        for(int i =0; i<hiddenOutputWeights.length; i++){
            //System.out.println("Hidden layer output " + hiddenLayerOutputs[i]);
            hiddenOutputWeights[i] = hiddenOutputWeights[i] - (learningRate * hiddenLayerOutputs[i] * deltaOutput);
        }
        //printMat(hiddenOutputWeights);
        //calculate deltas for every neuron in the hidden layer
        double[] hiddenDeltas = new double[hiddenLayerSize];
        for(int i=0; i<hiddenLayerSize; i++){
            //derivative for relu 
            double d = 0.0;
            if(hiddenLayerInputs[i] > 0){
                d = 1.0;
            }
            hiddenDeltas[i] = d * hiddenOutputWeights[i] * deltaOutput;
        }
        
        //update weights 
        for(int i =0; i<inputHiddenWeights[0].length;i++){
            for(int j=0; j< inputHiddenWeights.length; j++){
                inputHiddenWeights[j][i] = inputHiddenWeights[j][i] - (learningRate * input[i] * hiddenDeltas[i]);
            }
        }
        
        //printMat(inputHiddenWeights);
    }
    
    private void printMat(double[] mat){
        for(int i=0; i<mat.length; i++){
            System.out.print(mat[i] + " ");
        }
        System.out.println("");
    }
    public static void printMat(double[][] mat){
        for(int i =0; i<mat.length; i++){
            for(int j=0; j<mat[i].length; j++){
                System.out.print(mat[i][j] + " ");
            }
            System.out.println("----");
        }
    }
}
