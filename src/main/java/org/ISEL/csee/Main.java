package org.ISEL.csee;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {

        CSVHandler csvHandler = new CSVHandler("Admission_Predict.csv");
        List<List<String>> data = csvHandler.getCsvData();

        List<Double> toeflScore = new ArrayList<>();
        List<List<Double>> wholeData = new ArrayList<>();
        List<Double> admitRate = new ArrayList<>();
        // 첫 줄이 헤더인 경우, 1부터 시작
        for (int i = 1; i < data.size(); i++) {
            List<String> row = data.get(i);
            toeflScore.add(Double.parseDouble(row.get(2)) / 120.0);   // TOEFL Score
            admitRate.add(Double.valueOf(row.get(8)));    // Chance of Admit

            List<Double> oneRow = new ArrayList<>();
            for (int j = 1; j < row.size()-1; j++) {
                double val = Double.parseDouble(row.get(j));
                if(j == 1) val /= 340.0;
                else if(j == 2) val /= 120.0;
                else if(j == 3) val /= 5.0;
                else if(j == 4) val /= 5.0;
                else if(j == 5) val /= 5.0;
                else if(j == 6) val /= 10.0;
                else if(j == 7) val /= 1.0;
                oneRow.add(val);
            }
            oneRow.add(1.0);
            wholeData.add(oneRow);
        }

        // 각 행마다 TOEFL, Admit Rate 출력
//        for (int i = 0; i < toeflScore.size(); i++) {
//            System.out.println("TOEFL: " + toeflScore.get(i) + " | Admit Rate: " + admitRate.get(i));
//        }
        // 모든 data 출력
//        for(int i = 0; i < wholeData.size(); i++){
//            for(int j = 0; j < wholeData.get(i).size(); j++){
//                System.out.print(wholeData.get(i).get(j) + " ");
//            }
//            System.out.println();
//        }

        Utils utils = new Utils();
        float learningRate = 0.1f;
        float w = 0f;
        float b = 0f;
        int epochs = 1000001;
        List<Double> weights = new ArrayList<>();

        int featureSize = wholeData.get(0).size();

        for(int i = 0; i < featureSize; i++){
            weights.add(Math.random());
        }


//        utils.startGradientDescent(toeflScore, admitRate, learningRate, w, b, epochs);
        utils.startMultiGradientDescent(wholeData, admitRate, learningRate, weights, epochs);
    }
}