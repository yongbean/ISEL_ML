package org.ISEL.csee;

import java.io.*;
import java.util.*;

public class CSVHandler {
    private List<List<String>> csvData;

    public CSVHandler(String fileName) throws IOException {
        this.csvData = readCSV(fileName);
    }

    public List<List<String>> getCsvData() {
        return csvData;
    }

    // Read CSV from resources
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
}
