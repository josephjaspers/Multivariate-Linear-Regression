/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gradientdescent;

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

        initializeTheta();
    }

    public void initializeTheta() {
        for (int i = 0; i < constantArray.length; ++i) {
            constantArray[i] = 10;
        }
    }

    public void updateConstants() {
        for (int i = 0; i < trainingData.size(); ++i) {

            //System.out.println("\niteration " + i); 
            for (int j = 0; j < numbFeats; ++j) {
                costFunction(j, i);
            }
        }
    }
    //Cost function:  theta(j) = theta(j) + learningRate(y^(i) - H(theta)(x^(i),j) * (x^(i),j)
    //  H = hypothesis, ^(i) = an index (not pow of), j = feature index  
    //http://cs229.stanford.edu/notes/cs229-notes1.pdf (The algorithm is on pg. 5)
    public double costFunction(int j, int i) { //cost function for single update 
        int m = trainingData.size();
        double constant = constantArray[j];

        double sigma = 0;
        sigma += (y(i) - HypothesisTheta(i)) * x(i, j);

        System.out.println("index: " + i + " constant: " + j
                + " :  (" + y(i) + " - " + HypothesisTheta(i) + ") * " + x(i, j) + " = " + sigma);
        sigma *= learningRate;
        constant += sigma;

        constantArray[j] = constant;
        return constant;
    }

    public double HypothesisTheta(int index) {
        double total = 0;
        for (int i = 0; i < numbFeats; ++i) {
            total += trainingData.get(index)[i] * constantArray[i];
        }
        return total;
    }

    public double y(int index) {
        return trainingData.get(index)[numbFeats]; //Get the last element in trainingData which is the learningData
    }

    public double getY(int index) {
        return trainingData.get(index)[numbFeats];
    }

    public double x(int index, int jIndex) {
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
        if (cacheUpdated) {
            updateConstants();
        }

        if (data.length != numbFeats) {
            throw new IllegalArgumentException("invalid number of features");
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

    public String getConstantString() {
        String constants = "";

        for (int i = 0; i < constantArray.length; ++i) {
            constants += constantArray[i];
        }
        return constants;
    }

    public static void main(String[] args) {
        gradientDescent gd = new gradientDescent(new String[]{"F1", "F2", "F3"});
        double f1;
        double f2;
        double f3;
        double ans;
        for (int i = 0; i < 40000; i++) {
            f1 = Math.random() * 10 + 10;
            f2 = Math.random() * 10 + 10;
            f3 = Math.random() * 10 + 10;
            ans = f1 * 8 + f2 * 2 + f3 * 10;

            gd.addLearningData(new double[]{f1, f2, f3, ans});
        }
        System.out.println("\n \n");
        
        
        System.out.println("This test involves three variables, with respective constants - 8, 2, 10");
        
        System.out.println("\nThe constants were evaluated to be " + gd.getConstantString());
        System.out.println("Testing constants 6, 5, 2, \nevaluates to: " + gd.evaluate(new double[]{1, 2, 3}));
    }
}
