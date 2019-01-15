package test.dl.pojo;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Sumit Shrestha on 1/13/2019.
 */
public class Derivatives {
    private INDArray dw, db, cost;

    public Derivatives(INDArray dw, INDArray db, INDArray cost) {
        this.dw = dw;
        this.db = db;
        this.cost = cost;
    }

    public INDArray getDw() {
        return dw;
    }

    public INDArray getDb() {
        return db;
    }

    public INDArray getCost() {
        return cost;
    }

    @Override
    public String toString() {
        return "Derivatives{" +
                "dw=" + dw.toString() +
                ", db=" + db.toString() +
                '}';
    }
}
