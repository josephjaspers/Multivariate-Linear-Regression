/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gradientdescent;

import static gradientdescent.testClass.GradientDescentTest;
import static gradientdescent.testClass.network;
import java.util.ArrayList;

/**
 *
 * @author Joseph
 */
public class gradientDescent {

    private double tolerance = 1 / 10E10;
    private boolean converged = false; // Need to add convergence check
    private boolean regularization = false;
    private double learningRate = 0.003;
    private double regularizationParameter = 0.003;

    private double[] constants;
    private ArrayList<double[]> trainingData;  // the last double element is the Y of each Theta 
    private String[] featureNames;
    private int numbFeats;

    //boolean cacheUpdated = false;
    public gradientDescent(String[] featureNames) {
        this.featureNames = featureNames;
        numbFeats = featureNames.length + 1; //y(last feature is the y intercept)
        constants = new double[featureNames.length + 1]; //last feature y intercept
        trainingData = new ArrayList<>();
    }

    public void reTrain() {
        for (int i = 0; i < trainingData.size() && !converged; ++i) {
            for (int j = 0; j < numbFeats; ++j) {
                costFunction(j, i);
            }
        }
    } // reviews all the Data 

    public void reTrain(int times) {
        for (int i = 0; i < times; ++i) {
            reTrain();
        }
    }

    private void updateTrainingData(double[] data) {
        for (int j = 0; j < numbFeats; ++j) {
            costFunction(j, trainingData.size() - 1);
        }
    }

    //Cost function:  theta(j) = theta(j) + learningRate(y^(i) - H(theta)(x^(i),j) * (x^(i),j)
    //  H = hypothesis, ^(i) = an index (not pow of), j = feature index  
    //http://cs229.stanford.edu/notes/cs229-notes1.pdf (The algorithm is on pg. 5)
    private double costFunction(int j, int i) { //cost function for single update 
        int m = trainingData.size();
        double constant = constants[j];

        double sigma = 0;
        sigma += (y(i) - HypothesisTheta(i)) * x(i, j);
        //sigma += -(HypothesisTheta(i) - y(i)) * x(i, j);
        sigma *= learningRate;
        constant *= regularization(j);
        constant += sigma;

        constants[j] = constant;
        return constant;
    }

    private double regularization(int j) {
        if (!regularization || j == numbFeats - 1) {
            return 1; //For not updating the Y intercept
        }
        double m = trainingData.size();

        double lambda = regularizationParameter;
        return 1 - learningRate * (lambda / m);
    }

    private double HypothesisTheta(int index) {
        double total = 0;
        for (int i = 0; i < numbFeats; ++i) {
            if (i == numbFeats - 1) {
                total += constants[i]; //Y intercept
            } else {
                total += trainingData.get(index)[i] * constants[i];
            }
        }
        return total;
    }

    private double y(int index) {
        return trainingData.get(index)[numbFeats - 1]; //Get the last element in trainingData which is the learningData
    }

    private double x(int index, int jIndex) {
        return trainingData.get(index)[jIndex];
    }

    public void addLearningData(double[] data) {
        if (data.length != numbFeats) {
            throw new IllegalArgumentException("Must have all features + solution - curr sz: " + data.length);
        }
//        cacheUpdated = false;
        trainingData.add(data);
        updateTrainingData(data);
    }

    public void resetConstants() {
        constants = new double[constants.length];
    }

    public void resetConstants(boolean update) {
        resetConstants();
        if (update) {
            reTrain();
        }
    }

    public void clearTrainingData(boolean maintainConstants) {
        trainingData = new ArrayList<>();
        if (!maintainConstants) {
            resetConstants();
        }
    }

    public double evaluate(double[] data) {
        if (data.length != numbFeats - 1) {//Excludes y intercept
            throw new IllegalArgumentException("invalid number of features");
        }

        double total = 0;
        for (int i = 0; i < data.length; ++i) {
            total += data[i] * constants[i];
        }
        total += constants[numbFeats - 1]; //adds the y intercept 
        return total;
    }

    //turn on regularization - Good for data sets with a large amount of parameters 
    public boolean setRegularization(boolean onOff) {
        if (regularization == onOff) {
            return false;
        } else {
            regularization = onOff;
            resetConstants();
            reTrain();
            return true;
        }
    }

    public void setLearningRate(double lr) {
        learningRate = lr;
    }

    public void setTolerance(double t) {
        tolerance = t;
    }

    public void setRegularizationRate(double r) {
        regularizationParameter = r;
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getTolerance() {
        return tolerance;
    }

    public double[] getConstants() {
        return constants;
    }

    public int getFeatureLength() {
        return numbFeats;
    }

    public double getRegularizationParameter() {
        return regularizationParameter;
    }

    public ArrayList<double[]> getTrainingData() {
        return trainingData;
    }

    public boolean isRegularized() {
        return regularization;
    }

    public static void main(String[] args) {
        GradientDescentTest();
    }
}
