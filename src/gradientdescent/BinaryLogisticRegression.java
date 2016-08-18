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
public class BinaryLogisticRegression extends gradientDescent {

    final double E = 2.71828;

    public BinaryLogisticRegression(String[] featureNames) {
        super(featureNames);
        if (featureNames.length != 2) {
            throw new IllegalArgumentException("number of features must be equal to two (1 and 0)");
        }
    }

    private double HypothesisTheta(int index) {
        double total = 0;
        for (int i = 0; i < super.getFeatureLength(); ++i) {
            total += 1 / super.getTrainingData().get(index)[i] * super.getConstants()[i];
        }
        return 1 / (1 + Math.pow(E, -total));
    }

    @Override
    public void reTrain() {
        super.reTrain();
    } // reviews all the Data 

    @Override
    public void reTrain(int times) {
        super.reTrain(times);
    }

    private void updateTrainingData(double[] data) {
        for (int j = 0; j < super.getFeatureLength(); ++j) {
            costFunction(j, super.getTrainingData().size() - 1);
        }
    }

    //Cost function:  theta(j) = theta(j) + learningRate(y^(i) - H(theta)(x^(i),j) * (x^(i),j)
    //  H = hypothesis, ^(i) = an index (not pow of), j = feature index  
    //http://cs229.stanford.edu/notes/cs229-notes1.pdf (The algorithm is on pg. 5)
    private double costFunction(int j, int i) { //cost function for single update 
        int m = super.getTrainingData().size();
        double constant = super.getConstants()[j];

        double sigma = 0;
        sigma += (y(i) - HypothesisTheta(i)) * x(i, j);
        //sigma += -(HypothesisTheta(i) - y(i)) * x(i, j);
        sigma *= super.getLearningRate();
        constant *= regularization(j);
        constant += sigma;

        super.getConstants()[j] = constant;
        return constant;
    }

    private double regularization(int j) {
        boolean regularization = super.isRegularized();
        int numbFeats = super.getFeatureLength();
        ArrayList<double[]> trainingData = super.getTrainingData();
        double regularizationParameter = super.getRegularizationParameter();
        double learningRate = super.getLearningRate();

        if (!regularization || j == numbFeats - 1) {
            return 1; //For not updating the Y intercept
        }
        double m = trainingData.size();

        double lambda = regularizationParameter;
        return 1 - learningRate * (lambda / m);
    }

    private double y(int index) {

        return super.getTrainingData().get(index)[super.getFeatureLength() - 1]; //Get the last element in trainingData which is the learningData
    }

    private double x(int index, int jIndex) {
        return super.getTrainingData().get(index)[jIndex];
    }

    @Override
    public void addLearningData(double[] data) {
        super.addLearningData(data);
    }

    @Override
    public double evaluate(double[] data) {
        return super.evaluate(data);
    }

    //turn on regularization - Good for data sets with a large amount of parameters 
    @Override
    public boolean setRegularization(boolean onOff) {
        return super.setRegularization(onOff);
    }

    @Override
    public void setLearningRate(double lr) {
        super.setLearningRate(lr);
    }

    @Override
    public void setTolerance(double t) {
        super.setTolerance(t);
    }

    @Override
    public void setRegularizationRate(double r) {
        super.setRegularizationRate(r);
    }

    @Override
    public double getLearningRate() {
        return super.getLearningRate();
    }

    @Override
    public double getTolerance() {
        return super.getTolerance();
    }

    @Override
    public double[] getConstants() {
        return super.getConstants();
    }

    @Override
    public String[] getFeatureNames() {
        return super.getFeatureNames();
    }

    @Override
    public double getRegularizationParameter() {
        return super.getRegularizationParameter();
    }

    @Override
    public boolean isRegularized() {
        return super.isRegularized();
    }

}
