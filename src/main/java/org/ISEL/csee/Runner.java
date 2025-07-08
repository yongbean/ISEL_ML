package org.ISEL.csee;

import org.apache.commons.cli.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

// javac -cp ".;commons-cli-1.9.0.jar" org/ISEL/csee/*.java
// java -cp ".;commons-cli-1.9.0.jar" org.ISEL.csee.Runner -t train.csv -p test.csv -m multi

// or

//#!/bin/bash
//java -cp "runner.jar:commons-cli-1.5.0.jar" -jar runner.jar "$@"

// chmod +x runner

// ./runner -t train.csv -p test.csv -m multi


public class Runner {
    private String trainCsvFilePath;
    private String testCsvFilePath;
    private String modelType;

    public static void main(String[] args) throws IOException {
        Runner myRunner = new Runner();
        myRunner.run(args);
    }

    private void run(String[] args) throws IOException {
        Options options = createOptions();
        if (!parseOptions(options, args)) return;

        // 1. Load training data
        CSVHandler trainHandler = new CSVHandler(trainCsvFilePath);
        List<List<Double>> trainX = trainHandler.getNormalizedFeatures();
        List<Double> trainY = trainHandler.getLabels();

        // 2. Load test data
        CSVHandler testHandler = new CSVHandler(testCsvFilePath);
        List<List<Double>> testX = testHandler.getNormalizedFeatures();
        List<Double> testY = testHandler.getLabels();

        // 3. Hyperparameters
        float learningRate = 0.001f;
        int epochs = 100001;
        double threshold = 0.9;
        int tp = 0, fp = 0, fn = 0, tn = 0;
        double precision = 0.01f, recall = 0.01f, accuracy = 0.01f;

        int featureSize = trainX.getFirst().size();
        List<Double> weights = new ArrayList<>();
        for (int i = 0; i < featureSize; i++) weights.add(Math.random());
//        for (int i = 0; i < featureSize; i++) weights.add(0.0);

        // 4. Model Training & Prediction
        if (modelType.equalsIgnoreCase("single")) {
            System.out.println("▶ Training Single-Variable Linear Regression...");

            float w = 0f, b = 0f;
            // 2D to 1D
            List<Double> flatTrainX = flattenSingleFeature(trainX);
            List<Double> flatTestX = flattenSingleFeature(testX);

            SingleGradientDescent single = new SingleGradientDescent();
            float[] wb = single.run(flatTrainX, trainY, learningRate, w, b, epochs);

            System.out.println("\n▶ Predicting on test set:");
            for (int i = 0; i < flatTestX.size(); i++) {
                double x = flatTestX.get(i);
                double pred = wb[0] * x + wb[1];
                System.out.printf("Sample %d: Predicted = %.5f | Actual = %.5f%n", i, pred, testY.get(i));
            }

        } else if (modelType.equalsIgnoreCase("multi")) {
            System.out.println("▶ Training Multi-Variable Linear Regression...");

            MultiGradientDescent multi = new MultiGradientDescent();
            List<Double> learnedWeights = multi.run(trainX, trainY, learningRate, weights, epochs);

            System.out.println("\n▶ Predicting on test set:");
            for (int i = 0; i < testX.size(); i++) {
                double pred = 0;
                for (int j = 0; j < learnedWeights.size(); j++) {
                    pred += learnedWeights.get(j) * testX.get(i).get(j);
                }
                System.out.printf("Sample %d: Predicted = %.5f | Actual = %.5f%n", i, pred, testY.get(i));
            }

        } else if(modelType.equalsIgnoreCase("logistic")) {
            System.out.println("▶ Training single-feature Logistic Regression...");


            BinaryLogisticRegression logistic = new BinaryLogisticRegression();
            List<Double> logisticLearnedWeights = logistic.run(trainX, trainY, weights, learningRate, epochs);

            System.out.println("\n▶ Predicting on test set:");
            for (int i = 0; i < testX.size(); i++) {
                double pred = 0;
                for (int j = 0; j < logisticLearnedWeights.size(); j++) {
                    pred += logisticLearnedWeights.get(j) * testX.get(i).get(j);
                }
//                System.out.println("pred: " + pred + ", weight: " + logisticLearnedWeights.subList(0, logisticLearnedWeights.size()));

                // sigmoid 사용 후 theshold 값에 의해 0 or 1 지정
                double sig = logistic.sigmoid(pred);
                sig = (sig > threshold) ? 1:0;
                System.out.printf("Sample %d: Predicted = %.5f | Actual = %.5f%n", i, sig, testY.get(i));

                // tp, fn, fp, tn 값 지정
                if(testY.get(i) == 1) {
                    if(sig == 1) tp++;
                    else fn++;
                }
                else {
                    if(sig == 1) fp++;
                    else tn++;
                }
            }

            // calculate precision, recall, accuracy
            precision = (double) tp / (tp + fp);
            recall = (double) tp / (tp + fn);
            accuracy = (double) (tp + tn) / (tp + fn + fp + tn);

            System.out.println();
            System.out.println("tp = " + tp + " fp = " + fp + " fn = " + fn + " tn = " + tn);
            System.out.println("precision = " + precision + ", recall = " + recall + ", accuracy = " + accuracy);
            double f1 = 2 * precision * recall /(precision + recall);
            System.out.println("F1 score = " + f1);
            double result = (double) (tp + tn) / testY.size();
            System.out.println("Matched count: " + (tp + tn) + "/" + testY.size() + " (" + result + ")");
        } else if(modelType.equalsIgnoreCase("softmax")) {
            MultinomialLogisticRegression softmax = new MultinomialLogisticRegression();
            List<List<Double>> softmaxWeight = softmax.run(trainX, trainY, learningRate, epochs);

            System.out.println("\n▶ Predicting on test set:");

            int correct = 0;

            for (int i = 0; i < testX.size(); i++) {
                List<Double> xInstance = testX.get(i);
                List<Double> hx = softmax.predict(softmaxWeight, xInstance); // z = W * x
                List<Double> soft = softmax.softmax(hx); // softmax 확률 벡터

                // 예측 클래스 = 확률이 가장 높은 인덱스
                int predictedClass = 0;
                double maxProb = soft.get(0);
                for (int k = 1; k < soft.size(); k++) {
                    if (soft.get(k) > maxProb) {
                        maxProb = soft.get(k);
                        predictedClass = k;
                    }
                }

                int actualClass = testY.get(i).intValue();

                List<String> formattedProbs = new ArrayList<>();
                for (double prob : soft) {
                    formattedProbs.add(String.format("%.6f", prob));
                }
                String probString = "[" + String.join(", ", formattedProbs) + "]";

                System.out.printf("Sample %d: Predicted = %d | Actual = %d | Probabilities = %s%n",
                        i, predictedClass, actualClass, probString);

                if (predictedClass == actualClass) correct++;
            }

            double matched = (double) correct / testY.size();
            System.out.printf("▶ Accuracy = %.4f (%d/%d matched)\n", matched, correct, testY.size());

        }else {
            System.out.println("Invalid model type. Use 'single' or 'multi'.");
        }
    }

    private List<Double> flattenSingleFeature(List<List<Double>> X) {
        List<Double> result = new ArrayList<>();
        for (List<Double> row : X) {
            result.add(row.get(0)); // 단일 feature만 존재한다고 가정
        }
        return result;
    }

    private Options createOptions() {
        Options options = new Options();

        options.addOption(Option.builder("t").longOpt("train")
                .desc("train csv file")
                .hasArg()
                .argName("train csv file path(name)")
                .required()
                .build());

        options.addOption(Option.builder("p").longOpt("test")
                .desc("test csv file")
                .hasArg()
                .argName("test csv file path(name)")
                .required()
                .build());

        options.addOption(Option.builder("m").longOpt("model")
                .desc("model name (single or multi)")
                .hasArg()
                .argName("model name")
                .required()
                .build());

        return options;
    }

    private boolean parseOptions(Options options, String[] args) {
        CommandLineParser parser = new DefaultParser();
        try {
            CommandLine cmd = parser.parse(options, args);
            trainCsvFilePath = cmd.getOptionValue("t");
            testCsvFilePath = cmd.getOptionValue("p");
            modelType = cmd.getOptionValue("m");
        } catch (ParseException e) {
            printHelp(options);
            return false;
        }
        return true;
    }

    private void printHelp(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        String header = "ML Regression Model Runner";
        formatter.printHelp(header, options, true);
    }
}
