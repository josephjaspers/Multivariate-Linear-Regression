/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gradientdescent;

/**
 *
 * @author Joseph
 */
public class LogisticRegression extends gradientDescent{
    private final double E =  2.71828; 
    
    public LogisticRegression(String[] featureNames)  {
        super(featureNames);
    }
    @Override
    public double evaluate(double[] data) {
        double eval = super.evaluate(data); 
       // return 1 / (1 + Math.pow(E, eval));
       return eval;
    }
}
