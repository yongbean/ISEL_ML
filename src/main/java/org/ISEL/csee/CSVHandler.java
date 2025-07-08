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

        // 1. CSV 데이터에서 feature 부분 추출 (label 제외)
        List<List<Double>> rawFeatures = new ArrayList<>();
        for (int i = 1; i < rawCsvData.size(); i++) {  // 헤더 제외
            List<String> row = rawCsvData.get(i);
            List<Double> features = new ArrayList<>();
            for (int j = 0; j < row.size() - 1; j++) {  // 마지막 열은 label
                features.add(Double.parseDouble(row.get(j)));
            }
            rawFeatures.add(features);
        }

        if (rawFeatures.isEmpty()) return normalizedData;

        int numFeatures = rawFeatures.get(0).size();

        // 2. 각 feature별 min/max 계산
        double[] minVals = new double[numFeatures];
        double[] maxVals = new double[numFeatures];
        Arrays.fill(minVals, Double.POSITIVE_INFINITY);
        Arrays.fill(maxVals, Double.NEGATIVE_INFINITY);

        for (List<Double> row : rawFeatures) {
            for (int j = 0; j < numFeatures; j++) {
                double val = row.get(j);
                if (val < minVals[j]) minVals[j] = val;
                if (val > maxVals[j]) maxVals[j] = val;
            }
        }

        // 3. 정규화 + bias term 추가
        for (List<Double> row : rawFeatures) {
            List<Double> normRow = new ArrayList<>();
            for (int j = 0; j < numFeatures; j++) {
                double min = minVals[j];
                double max = maxVals[j];
                double val = row.get(j);
                double norm = (max - min == 0.0) ? 0.0 : (val - min) / (max - min);
                normRow.add(norm);
            }
            normRow.add(1.0);  // bias term
            normalizedData.add(normRow);
        }

        return normalizedData;
    }

    // 라벨(Chance of Admit) 반환
    private Map<String, Integer> labelToIndex = new HashMap<>();

    public List<Double> getLabels() {
        List<Double> labels = new ArrayList<>();
        int currentIndex = 0;

        for (int i = 1; i < rawCsvData.size(); i++) {
            List<String> row = rawCsvData.get(i);
            String labelStr = row.getLast().trim();

            try {
                // 숫자면 그대로 파싱
                labels.add(Double.parseDouble(labelStr));
            } catch (NumberFormatException e) {
                // 문자열이면 매핑
                if (!labelToIndex.containsKey(labelStr)) {
                    labelToIndex.put(labelStr, currentIndex++);
                }
                labels.add((double) labelToIndex.get(labelStr));
            }
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
