package test.dl.service;

import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import test.dl.pojo.Dataload;
import test.dl.pojo.Derivatives;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;
import static test.dl.util.MatrixOperations.log;
import static test.dl.util.MatrixOperations.sigmoid;
import static test.dl.util.MatrixOperations.sigmoid1;

/**
 * Created by Sumit Shrestha on 1/13/2019.
 */
@Service
public class LogisticRegressionService implements MLModel {
    private final DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
    private final int height = 250;
    private final int width = 250;
    private final int channels = 1;
    private final NativeImageLoader loader = new NativeImageLoader(height, width, channels);
    private final ImageLoader imageLoader = new ImageLoader();
    private Parameter parameter;

    @Autowired
    public LogisticRegressionService() {
//        train(2000, 0.005f);
    }

    @Override
    public Map train(int numIterations, float learningRate) {
        LinkedHashMap rs = new LinkedHashMap<>();
        try {
            long tik = System.currentTimeMillis();
            System.out.println("Training Logistic Regression model with " + numIterations + " iteratoins @" + learningRate);
            parameter = new Parameter();
//            Dataload train = loadDataset("C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\train\\Negatives", "C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\train\\George_W_Bush");
            Dataload train = loadDataset("C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\test\\negatives", "C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\test\\bush");
            if (!train.isSuccess()) {
                System.out.println("x was not init");
                rs.put("result", "failure");
                return rs;
            }
            final int[] xShape = train.getX().shape();

            parameter.w = Nd4j.zeros(xShape[0], 1);
            optimize(parameter, train.getX(), train.getY(), numIterations, learningRate);
//            optimize(parameter.w, parameter.b, train.getX(), train.getY(), numIterations, learningRate);
            double[] yPredictionTrain = predict(parameter.w, parameter.b, train.getX());
            printPrediction(train.getY(), yPredictionTrain);
            System.out.println("Done training. Took " + (System.currentTimeMillis() - tik) / 1000 + " seconds.");

            rs.put("result", "success");
            rs.put("parameters", parameter.getJSON());
            rs.put("trainPrediction", yPredictionTrain);
            return rs;
        } catch (IOException e) {
            e.printStackTrace();
            rs.put("result", e.getMessage());
            return rs;
        }

    }

    @Override
    public Map test() {
        LinkedHashMap rs = new LinkedHashMap<>();
        Dataload test = null;
        try {
            test = loadDataset("C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\test\\negatives", "C:\\Users\\sumit\\Projects\\DeepLearning\\Dataset\\lfw\\test\\bush");
            double[] yPredictionTest = predict(parameter.w, parameter.b, test.getX());
            printPrediction(test.getY(), yPredictionTest);
            rs.put("result", "success");
            rs.put("trainPrediction", yPredictionTest);
            return rs;
        } catch (IOException e) {
            e.printStackTrace();
            rs.put("result", e.getMessage());
            return rs;
        }
    }

    @Override
    public Map getParameters() {
        LinkedHashMap<Object, Object> map = new LinkedHashMap<>();
        map.put("result", "untrained model");
        return parameter == null ? map : parameter.getJSON();
    }

    private void printPrediction(INDArray y, double[] yPrediction) {
        INDArray prediction = Nd4j.mean(abs(Nd4j.create(yPrediction).sub(y).mul(100))).mul(-1).add(100);
        System.out.println("this is prediction :" + prediction.toString());
        System.out.println("this is y prediction :" + Arrays.toString(yPrediction));
        System.out.println("this is y :" + y.toString());
    }

    private Dataload loadDataset(String negativeDatasetFolder, String positiveDatasetFolder) throws IOException {
        long tik = System.currentTimeMillis();
        INDArray x = loadImageFromFolder(negativeDatasetFolder);
        int negativeCount = x.shape()[0];
        System.out.println("Negative images read. Count " + negativeCount + ". Took " + (System.currentTimeMillis() - tik) / 1000 + " seconds.");
        tik = System.currentTimeMillis();
        INDArray positiveImages = loadImageFromFolder(positiveDatasetFolder);
        int positiveCount = positiveImages.shape()[0];
        x = Nd4j.vstack(x, positiveImages);
        System.out.println("Positive images read. Count " + positiveCount + ". Took " + (System.currentTimeMillis() - tik) / 1000 + " seconds.");
        if (x == null) {
            System.out.println("x was not init");
            return new Dataload(false);
        }

        x = x.transpose();
        scalar.transform(x); // normalize in nd4j way

        INDArray y = Nd4j.zeros(1, negativeCount);
        y = Nd4j.hstack(y, Nd4j.ones(1, positiveCount));
        System.out.printf("this is Y: " + y.toString());

        return new Dataload(negativeCount, positiveCount, x, y, true);
    }

    private INDArray loadImageFromFolder(String path) throws IOException {
        File folder = new File(path);
        INDArray x = null;
        File[] files = folder.listFiles();
        for (File imageFile : files) {
            if (imageFile == null || !imageFile.exists() || !imageFile.getName().endsWith("jpg"))
                continue;
            INDArray imageArray = loader.asRowVector(imageFile);
            if (x == null) {
                x = imageArray;
            } else {
                x = Nd4j.vstack(x, imageArray);
            }
        }
        return x;
    }

    private Derivatives propagate(INDArray w, double b, INDArray X, INDArray Y) {
//        System.out.printf("dimensions: ");
//        System.out.println("w : " + Arrays.toString(w.shape()));
//        System.out.println("x : " + Arrays.toString(X.shape()));
//        System.out.println("y : " + Arrays.toString(Y.shape()));
        double m = X.shape()[1];
        // do sigmoid here by yourself
        INDArray z = w.transpose().mmul(X).add(b);
        INDArray a = sigmoid1(z);
//        INDArray a = Transforms.sigmoid(z);
//        INDArray a = sigmoid(z);
        System.out.println("this is max and min of a :" + a.maxNumber().toString() + " and " + a.minNumber().toString());
        INDArray cost = Nd4j.sum(Y.mul(log(a)).add(Y.sub(1.0d).mul(-1.0d).mul(log(a.sub(1).mul(-1))))).mul(-1.0d / m);
//        System.out.println("this is cost " + cost.toString());

        INDArray sub = a.sub(Y);
        INDArray mmul = X.mmul(sub.transpose());
        INDArray dw = mmul.mul(1.0d / m);
//        System.out.printf("this is values " + sub.toString() + " mmul " + mmul.toString() + " dw " + dw.toString());
        System.out.println("this is max value in mmul :" + mmul.maxNumber().toString());
        INDArray db = Nd4j.sum(a.sub(Y)).mul(1.0d / m);

        return new Derivatives(dw, db, cost);
    }

    private void optimize(Parameter p, INDArray X, INDArray Y, int numIterations, double learningRate) {
//    private void optimize(INDArray w, double b, INDArray X, INDArray Y, int numIterations, double learningRate) {

        for (int i = 0; i < numIterations; i++) {
            Derivatives d = propagate(p.w, p.b, X, Y);
            if (i % 30 == 0)
                System.out.println("this is cost after " + i + " iteration : " + d.getCost().toString());
//            System.out.println("this is derivate " + d.toString());
            INDArray mul = d.getDw().mul(learningRate);
            p.w = p.w.sub(mul);
//            System.out.println("this is learning rate " + learningRate);
            System.out.println("this is max value in dw :" + d.getDw().maxNumber().toString());
            System.out.println("this is mul " + mul.maxNumber().toString());
            System.out.println("this is w max num :" + p.w.maxNumber().toString());
            p.b = d.getDb().mul(learningRate * -1.0d).add(p.b).getDouble(0);
//            System.out.println("this is for b :" + .toString());
//            System.out.println("this is w :" + w.toString());
//            System.out.println("this is b " + b);
        }
    }

    private double[] predict(INDArray w, double b, INDArray X) {
        int m = X.shape()[1];

//        w = w.reshape(X.shape()[0], 1);
        INDArray z = w.transpose().mmul(X).add(b);
        System.out.println("min in z :" + z.minNumber().toString());
        System.out.println("min in w :" + w.minNumber().toString());
        INDArray a = sigmoid(z);
        System.out.println("this is min in a : " + a.minNumber().toString());
        System.out.println("this is max in a : " + a.maxNumber().toString());
        int i1 = a.shape()[1];
        double[] yPrediction = new double[i1];
        for (int i = 0; i < i1; i++) {
            yPrediction[i] = a.getDouble(0, i) > 0.5 ? 1 : 0;
        }

        return yPrediction;
    }

    public class Parameter {
        private INDArray w;
        private double b = 0;

        public void printValues() {
            System.out.println("for parameters: ");
            System.out.println("w: " + w.toString());
            System.out.println("b: " + b);
        }

        public INDArray getW() {
            return w;
        }

        public void setW(INDArray w) {
            this.w = w;
        }

        public double getB() {
            return b;
        }

        public void setB(double b) {
            this.b = b;
        }

        public Map getJSON() {
            LinkedHashMap<Object, Object> parameter = new LinkedHashMap<>();
            parameter.put("w", w.toString());
            parameter.put("b", b);
            return parameter;
        }
    }
}
