package test.dl.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by Sumit Shrestha on 1/13/2019.
 */
public class MatrixOperations {
    public static INDArray sigmoid(INDArray z) {
        return Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Sigmoid(z));
    }

    public static INDArray log(INDArray z) {
        return Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Log(z));
    }

    public static INDArray add(INDArray a, INDArray b) {
        return Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(a, b, a, 1));
    }

    public static INDArray sigmoid1(INDArray z) {
        return Nd4j.ones(1, z.shape()[1]).div(Transforms.exp(z.mul(-1)).add(1));
    }
}
