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
public class TestClass {

    public static void testGradientDescent() {
        gradientDescent gd = new gradientDescent(new String[]{"F1", "F2", "F3"});
        double c1 = Math.random() * 10 % 5;
        double c2 = Math.random() * 10 % 5;
        double c3 = Math.random() * 10 % 5;

        double f1;
        double f2;
        double f3;
        double ans;

        for (int i = 0; i < 4000; i++) {
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

    public static void testLogisticRegression() {
        LogisticRegression gd = new LogisticRegression(new String[]{"F1", "F2", "F3"});

        double c1 = Math.random() * 10 % 10;
        double c2 = Math.random() * 10 % 10;
        double c3 = Math.random() * 10 % 10;

        double f1;
        double f2;
        double f3;
        double ans;

        for (int i = 0; i < 4000; i++) {
            f1 = Math.random() * 10 + 10;
            f2 = Math.random() * 10 + 10;
            f3 = Math.random() * 10 + 10;

            if (f1 * c1 + f2* c2 + f3* c3 > 200) {
                ans = 1;
            } else {
                ans = 0;
            }

            gd.addLearningData(new double[]{f1, f2, f3, ans});
        }
        f1 = Math.random() * 10 + 10;
        f2 = Math.random() * 10 + 10;
        f3 = Math.random() * 10 + 10;

        System.out.println("This test involves three variables, with respective constants - " + c1 + " " + c2 + " " + c3);
        System.out.println("Testing parameters " + f1 + " " + f2 + " " + f3 + " " + "\nevaluates to: " + gd.evaluate(new double[]{f1, f2, f3}) + "\n");

        double total = 0;
        double[] constants = gd.getConstants();
        total += c1 * constants[0];
        total += c2 * constants[1];
        total += c3 * constants[2];
        System.out.println("the evaluated sum before logistic 1 / 1 - e ^ theta transpose x is " + total);

        System.out.println("The constants were evaluated to be ");

        double[] theConst = gd.getConstants();
        for (int i = 0; i < theConst.length; ++i) {
            System.out.println(theConst[i]);
        }

    }
}
