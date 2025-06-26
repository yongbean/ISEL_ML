package org.ISEL.csee;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

//TIP 코드를 <b>실행</b>하려면 <shortcut actionId="Run"/>을(를) 누르거나
// 에디터 여백에 있는 <icon src="AllIcons.Actions.Execute"/> 아이콘을 클릭하세요.
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
            for (int j = 0; j < row.size()-1; j++) {
                oneRow.add(Double.parseDouble(row.get(j)));
            }
            wholeData.add(oneRow);
        }

        // 각 행마다 TOEFL, Admit Rate 출력
//        for (int i = 0; i < toeflScore.size(); i++) {
//            System.out.println("TOEFL: " + toeflScore.get(i) + " | Admit Rate: " + admitRate.get(i));
//        }

        Utils utils = new Utils();
        float learningRate = 0.01f;
        float w = 0f;
        float b = 0f;
        int epochs = 100000;

        utils.startGradientDescent(toeflScore, admitRate, learningRate, w, b, epochs);
        utils.startMultiGradientDescent(wholeData, admitRate, learningRate, w, b, epochs);
    }
}