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

            int featureSize = trainX.get(0).size();
            List<Double> weights = new ArrayList<>();
            for (int i = 0; i < featureSize; i++) weights.add(Math.random());

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

            int featureSize = trainX.get(0).size();
            List<Double> weights = new ArrayList<>();
//            for (int i = 0; i < featureSize; i++) weights.add(Math.random());
            for (int i = 0; i < featureSize; i++) weights.add(0.0);

            LogisticRegression logistic = new LogisticRegression();
            List<Double> logisticLearnedWeights = logistic.run(trainX, trainY, weights, learningRate, epochs);

            System.out.println("\n▶ Predicting on test set:");
            int count = 0;
            for (int i = 0; i < testX.size(); i++) {
                double pred = 0;
                for (int j = 0; j < logisticLearnedWeights.size(); j++) {
                    pred += logisticLearnedWeights.get(j) * testX.get(i).get(j);
                }
                System.out.println("pred: " + pred + ", weight: " + logisticLearnedWeights.subList(0, logisticLearnedWeights.size()));
                double sig = logistic.sigmoid(pred);
                sig = (sig > 0.9) ? 1:0;
                System.out.printf("Sample %d: Predicted = %.5f | Actual = %.5f%n", i, sig, testY.get(i));
                if((int)sig == testY.get(i)) count++;
            }
            System.out.println("Matched count: " + count + "/" + testY.size());
        } else {
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
