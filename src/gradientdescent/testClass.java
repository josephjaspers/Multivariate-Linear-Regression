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
public class testClass {

    public static void GradientDescentTest() {
        gradientDescent gd = new gradientDescent(new String[]{"F1", "F2", "F3"});
        double c1 = Math.random() * 10 + 10;
        double c2 = Math.random() * 10 + 10;
        double c3 = Math.random() * 10 + 10;

        double f1;
        double f2;
        double f3;
        double ans;

        for (int i = 0; i < 10000; i++) {
            f1 = Math.random() * 10 + 10;
            f2 = Math.random() * 10 + 10;
            f3 = Math.random() * 10 + 10;
            ans = f1 * c1 + f2 * c2 + f3 * c3;
            gd.addLearningData(new double[]{f1, f2, f3, ans});
        }
        f1 = Math.random() * 10 + 10;
        f2 = Math.random() * 10 + 10;
        f3 = Math.random() * 10 + 10;

        gd.reTrain(10); //reTrain the data looks over the given dataSet and reiterates gradient Descent upon it 

        System.out.println("This test involves three variables, with respective constants - " + c1 + " " + c2 + " " + c3);
        System.out.println("Testing parameters " + f1 + " " + f2 + " " + f3 + " " + "\nevaluates to: " + gd.evaluate(new double[]{f1, f2, f3}) + "\n");
        System.out.println("The constants were evaluated to be ");

        double[] theConst = gd.getConstants();
        for (int i = 0; i < theConst.length; ++i) {
            System.out.println(theConst[i]);
        }
    }
}
