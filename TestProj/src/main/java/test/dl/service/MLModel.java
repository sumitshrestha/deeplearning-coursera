package test.dl.service;

import java.util.Map;

/**
 * Created by Sumit Shrestha on 1/13/2019.
 */
public interface MLModel {
    Map train(int numIterations, float learningRate);

    Map test();

    Map getParameters();
}
