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

    double tolerance = 1 / 10E9;
    boolean converged = false;
    boolean regularization = false;
    double learningRate = 0.003;
    double regularizationParameter = 0.003;

    double[] constants;
    ArrayList<double[]> trainingData;  // the last double element is the Y of each Theta 
    String[] featureNames;
    int numbFeats;

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

    public double evaluate(double[] data) {
        if (data.length != numbFeats - 1) {//Excludes y intercept
            throw new IllegalArgumentException("invalid number of features");
        }

        double total = 0;
        for (int i = 0; i < data.length; ++i) {
            total += data[i] * constants[i];
        }
        return total;
    }

    //turn on regularization - Good for data sets with a large amount of parameters 
    public boolean setRegularization(boolean onOff) {
        if (regularization == onOff) {
            return false;
        } else {
            regularization = onOff;
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

    public double getLearningRate() {
        return learningRate;
    }

    public double getTolerance() {
        return tolerance;
    }

    public double[] getConstants() {
        return constants.clone();
    }

    public double getRegularizationParameter() {
        return regularizationParameter;
    }

    public boolean isRegularized() {
        return regularization;
    }

    public static void main(String[] args) {
        gradientDescent gd = new gradientDescent(new String[]{"F1", "F2", "F3"});
        //BinaryLogisticRegression gd = new BinaryLogisticRegression(new String[]{"F1", "F2", "F3"});
        double c1 = Math.random() * 10 % 5;
        double c2 = Math.random() * 10 % 5;
        double c3 = Math.random() * 10 % 5;

        double f1;
        double f2;
        double f3;
        double ans;

        for (int i = 0; i < 100; i++) {
            f1 = Math.random() * 10 + 10;
            f2 = Math.random() * 10 + 10;
            f3 = Math.random() * 10 + 10;
            ans = f1 * c1 + f2 * c2 + f3 * c3;

            //For logistic Regression test
//            if (ans > 50) {
//                ans = 1;
//            } else {
//                ans = 0;
//            }
            gd.addLearningData(new double[]{f1, f2, f3, ans});
        }
        f1 = Math.random() * 10 + 10;
        f2 = Math.random() * 10 + 10;
        f3 = Math.random() * 10 + 10;

        gd.reTrain(100); //reTrain the data looks over the given dataSet and reiterates gradient Descent upon it 

        System.out.println("This test involves three variables, with respective constants - " + c1 + " " + c2 + " " + c3);
        System.out.println("Testing parameters " + f1 + " " + f2 + " " + f3 + " " + "\nevaluates to: " + gd.evaluate(new double[]{f1, f2, f3}) + "\n");
        System.out.println("The constants were evaluated to be ");

        double[] theConst = gd.getConstants();
        for (int i = 0; i < theConst.length; ++i) {
            System.out.println(theConst[i]);
        }
    }
}
