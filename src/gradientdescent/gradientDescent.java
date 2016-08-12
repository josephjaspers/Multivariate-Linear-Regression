/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gradientdescent;

import static gradientdescent.TestClass.testLogisticRegression;
import java.util.ArrayList;

/**
 *
 * @author Joseph
 */
public class gradientDescent {

    double learningRate = 0.0003;

    double[] constantArray;
    ArrayList<double[]> trainingData;  // the last double element is the Y of each Theta 
    String[] featureNames;
    int numbFeats;

    boolean cacheUpdated = false;

    public gradientDescent(String[] featureNames) {
        this.featureNames = featureNames;
        numbFeats = featureNames.length;
        constantArray = new double[featureNames.length];
        trainingData = new ArrayList<>();
    }

    private void updateConstants() {
        for (int i = 0; i < trainingData.size(); ++i) {
            for (int j = 0; j < numbFeats; ++j) {
                costFunction(j, i);
            }
        }
    }

    //Cost function:  theta(j) = theta(j) + learningRate(y^(i) - H(theta)(x^(i),j) * (x^(i),j)
    //  H = hypothesis, ^(i) = an index (not pow of), j = feature index  
    //http://cs229.stanford.edu/notes/cs229-notes1.pdf (The algorithm is on pg. 5)
    private double costFunction(int j, int i) { //cost function for single update 
        int m = trainingData.size();
        double constant = constantArray[j];

        double sigma = 0;
        sigma += (y(i) - HypothesisTheta(i)) * x(i, j);
        //System.out.println(HypothesisTheta(i));
        //System.out.println("index: " + i + " constant: " + j
        //        + " :  (" + y(i) + " - " + HypothesisTheta(i) + ") * " + x(i, j) + " = " + sigma);
        sigma *= learningRate;
        constant += sigma;

        constantArray[j] = constant;
        return constant;
    }

    private double HypothesisTheta(int index) {
        double total = 0;
        for (int i = 0; i < numbFeats; ++i) {
            total += trainingData.get(index)[i] * constantArray[i];
        }
        return total;
    }

    private double y(int index) {
        return trainingData.get(index)[numbFeats]; //Get the last element in trainingData which is the learningData
    }

    private double x(int index, int jIndex) {
        return trainingData.get(index)[jIndex];
    }

    public void addLearningData(double[] data) {
        if (data.length != numbFeats + 1) {
            throw new IllegalArgumentException("Must have all features + solution");
        }
        cacheUpdated = true;
        trainingData.add(data);
    }

    public double evaluate(double[] data) {
        if (data.length != numbFeats) {
            throw new IllegalArgumentException("invalid number of features");
        }

        if (cacheUpdated) {
            updateConstants();
        }

        double total = 0;
        for (int i = 0; i < data.length; ++i) {
            total += data[i] * constantArray[i];
        }
        return total;
    }

    public double[] getConstants() {
        return constantArray;
    }
    public static void main(String[] args) {
        testLogisticRegression();
    }
}
