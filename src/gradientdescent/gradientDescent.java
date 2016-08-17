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

    int minimumIterations;  
    double tolerance = 0.000000001;
    boolean converged = false;
    //Convergence check not yet supported 
    double learningRate = 0.003;

    double[] constants;
    ArrayList<double[]> trainingData;  // the last double element is the Y of each Theta 
    String[] featureNames;
    int numbFeats;

    boolean cacheUpdated = false;

    public gradientDescent(String[] featureNames) {
        this.featureNames = featureNames;
        numbFeats = featureNames.length + 1; //y(last feature is the y intercept)
        constants = new double[featureNames.length + 1]; //last feature y intercept
        trainingData = new ArrayList<>();
    }

    public void updateConstants() {
        for (int i = 0; i < trainingData.size() && !converged; ++i) {
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
        double constant = constants[j];

        double sigma = 0;
        sigma += (y(i) - HypothesisTheta(i)) * x(i, j);
        sigma *= learningRate;
        constant += sigma;

        constants[j] = constant;
        return constant;
    }

    public double HypothesisTheta(int index) {
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

    public double y(int index) {
        return trainingData.get(index)[numbFeats - 1]; //Get the last element in trainingData which is the learningData
    }

    public double x(int index, int jIndex) {
        return trainingData.get(index)[jIndex];
    }

    public void addLearningData(double[] data) {
        if (data.length != numbFeats) {
            throw new IllegalArgumentException("Must have all features + solution - curr sz: " + data.length);
        }
        cacheUpdated = true;
        trainingData.add(data);
    }

    public double evaluate(double[] data) {
        if (data.length != numbFeats - 1) {//Excludes y intercept
            throw new IllegalArgumentException("invalid number of features");
        }

        if (cacheUpdated) {
            updateConstants();
        }

        double total = 0;
        for (int i = 0; i < data.length; ++i) {
            total += data[i] * constants[i];
        }
        return total;
    }

    public double[] getConstants() {
        return constants;
    }

    public static void main(String[] args) {
        gradientDescent gd = new gradientDescent(new String[]{"F1", "F2", "F3"});
        double c1 = Math.random() * 10 % 5;
        double c2 = Math.random() * 10 % 5;
        double c3 = Math.random() * 10 % 5;

        double f1;
        double f2;
        double f3;
        double ans;

        for (int i = 0; i < 1000; i++) {
            f1 = Math.random() * 10 + 10;
            f2 = Math.random() * 10 + 10;
            f3 = Math.random() * 10 + 10;
            ans = f1 * c1 + f2 * c2 + f3 * c3;

            gd.addLearningData(new double[]{f1, f2, f3, ans});
        }
        f1 = Math.random() * 10 + 10;
        f2 = Math.random() * 10 + 10;
        f3 = Math.random() * 10 + 10;

        System.out.println("This test involves three variables, with respective constants - " + c1 + " " + c2 + " " + c3);
        System.out.println("Testing parameters " + f1 + " " + f2 + " " + f3 + " " + "\nevaluates to: " + gd.evaluate(new double[]{f1, f2, f3}) + "\n");
        System.out.println("The constants were evaluated to be ");

        double[] theConst = gd.getConstants();
        for (int i = 0; i < theConst.length; ++i) {
            System.out.println(theConst[i]);
        }
    }
}
