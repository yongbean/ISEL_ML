package org.ISEL.csee;

import java.io.*;
import java.util.*;

public class CSVHandler {
    private final List<List<String>> rawCsvData;

    public CSVHandler(String fileName) throws IOException {
        this.rawCsvData = readCSV(fileName);
    }

    // 원시 데이터 접근 (원하면 여전히 사용 가능)
    public List<List<String>> getCsvData() {
        return rawCsvData;
    }

    // 정규화된 feature 행렬 (wholeData) 반환
    public List<List<Double>> getNormalizedFeatures() {
        List<List<Double>> normalizedData = new ArrayList<>();

        // 첫 줄이 헤더인 경우, 1부터 시작
        for (int i = 1; i < rawCsvData.size(); i++) {
            List<String> row = rawCsvData.get(i);
            List<Double> oneRow = new ArrayList<>();

            for (int j = 0; j < row.size() - 1; j++) {
                double val = Double.parseDouble(row.get(j));
                if (j == 0) val /= 340.0;     // GRE
                else if (j == 1) val /= 120.0; // TOEFL
                else if (j == 2) val /= 5.0;   // University Rating
                else if (j == 3) val /= 5.0;   // SOP
                else if (j == 4) val /= 5.0;   // LOR
                else if (j == 5) val /= 10.0;  // CGPA
                else if (j == 6) val /= 1.0;   // Research
                oneRow.add(val);
            }
            oneRow.add(1.0); // bias term
            normalizedData.add(oneRow);
        }

        return normalizedData;
    }

    // 라벨(Chance of Admit) 반환
    public List<Double> getLabels() {
        List<Double> labels = new ArrayList<>();
        for (int i = 1; i < rawCsvData.size(); i++) {
            List<String> row = rawCsvData.get(i);
            labels.add(Double.parseDouble(row.get(7)));
        }
        return labels;
    }

    // Optional: write to disk
    public void writeCSV(List<List<String>> csvFileList, String fileName) throws IOException {
        String fileLocation = System.getProperty("user.dir") + File.separator + fileName;
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileLocation))) {
            for (List<String> row : csvFileList) {
                bw.write(String.join(",", row));
                bw.newLine();
            }
        }
    }

    // 내부: CSV 파일 읽기
    private List<List<String>> readCSV(String fileName) throws IOException {
        List<List<String>> csvFileList = new ArrayList<>();
        InputStream is = getClass().getClassLoader().getResourceAsStream(fileName);

        if (is == null) {
            throw new FileNotFoundException("Resource file not found: " + fileName);
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                List<String> aLine = Arrays.asList(line.split(","));
                csvFileList.add(aLine);
            }
        }

        return csvFileList;
    }
}
