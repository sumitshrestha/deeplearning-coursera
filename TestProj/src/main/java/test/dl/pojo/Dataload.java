package test.dl.pojo;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Sumit Shrestha on 1/13/2019.
 */
public class Dataload {
    private int negativeDataCount, positiveDataCount;
    private INDArray x, y;
    private boolean success;

    public Dataload(int negativeDataCount, int positiveDataCount, INDArray x, INDArray y, boolean success) {
        this.negativeDataCount = negativeDataCount;
        this.positiveDataCount = positiveDataCount;
        this.x = x;
        this.y = y;
        this.success = success;
    }

    public Dataload(boolean success) {
        this.success = success;
    }

    public int getNegativeDataCount() {
        return negativeDataCount;
    }

    public int getPositiveDataCount() {
        return positiveDataCount;
    }

    public INDArray getX() {
        return x;
    }

    public INDArray getY() {
        return y;
    }

    public boolean isSuccess() {
        return success;
    }
}
