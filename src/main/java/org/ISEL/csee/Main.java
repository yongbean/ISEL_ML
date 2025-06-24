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

        List<String> TOEFLScore = new ArrayList<String>();
        List<String> admitRate = new ArrayList<>();
        // 첫 줄이 헤더인 경우, 1부터 시작
        for (int i = 1; i < data.size(); i++) {
            List<String> row = data.get(i);
            TOEFLScore.add(row.get(2));   // TOEFL Score
            admitRate.add(row.get(8));    // Chance of Admit
        }

        // 각 행마다 TOEFL, Admit Rate 출력
        for (int i = 0; i < TOEFLScore.size(); i++) {
            System.out.println("TOEFL: " + TOEFLScore.get(i) + " | Admit Rate: " + admitRate.get(i));
        }
    }
}