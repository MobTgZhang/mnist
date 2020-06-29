import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class LeNetMNIST {
    private static final Logger log = LoggerFactory.getLogger(LeNetMNIST.class);
    public static void main(String[] args) throws IOException {
        int nChannels = 1;
        int outputNum = 10;
        int batchsize = 64;
        int nEpoches = 5;
        int seed = 123;
        System.out.println("Loading data...");
        log.info("Loading data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchsize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchsize,false,12345);
        log.info("Build model ...");
        System.out.println("Build model ...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        log.info("Training model ...");
        System.out.println("Training model ...");
        model.setListeners(new ScoreIterationListener(10),
                new EvaluativeListener(mnistTest,1, InvocationType.EPOCH_END));
        model.fit(mnistTrain,nEpoches);
        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"lenetmnist.zip");
        log.info("Saving model to tmp folder:"+path);
        System.out.println("Saving model to tmp folder:"+path);
        model.save(new File(path),true);
        log.info("The traing end!");
        System.out.println("The traing end!");
    }
}
