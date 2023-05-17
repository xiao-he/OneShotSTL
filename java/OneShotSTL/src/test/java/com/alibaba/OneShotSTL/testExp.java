package com.alibaba.OneShotSTL;

import java.io.*;
import org.junit.Test;
import java.util.*;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

public class testExp {
    @Test
    public void testSyn1() throws PythonExecutionException, IOException {
        String inputFile = "/Users/xiaohe/Projects/OneShotSTL-supp/data/syn1.json";
        JSonLoader loader = new JSonLoader();
        Map<String, Object> data = loader.load(inputFile);
        int T = (int) data.get("period");
        int trainTestSplit = (int) data.get("trainTestSplit");
        List<Double> y = (List<Double>) data.get("ts");
        List<Double> trend = (List<Double>) data.get("trend");
        List<Double> seasonal = (List<Double>) data.get("seasonal");
        List<Double> residual = (List<Double>) data.get("residual");
        List<Integer> timestamp = new ArrayList<>();
        for (int i = 0; i < y.size(); i++) {
            timestamp.add(i);
        }

        OneShotSTLAlg model = new OneShotSTLAlg(T, 10, 10000, 1.0, 0.5, 1.0, 5, "LS", "LAD", "LS", 1);
        List<Double> initY = y.subList(0, trainTestSplit);
        Map<String, double[]> res = model.initialize(initY);
//        List<Double> decomposedTrend = new ArrayList<>();
//        List<Double> decomposedSeasonal = new ArrayList<>();
//        List<Integer> decomposedTimestamp = new ArrayList<>();
//        for (int i = 0; i < res.get("trend").length; i++) {
//            decomposedTrend.add(res.get("trend")[i]);
//            decomposedSeasonal.add(res.get("seasonal")[i]);
//            decomposedTimestamp.add(i);
//        }
        List<Double> decomposedTrend = new ArrayList<>();
        List<Double> decomposedSeasonal = new ArrayList<>();
        List<Integer> decomposedTimestamp = new ArrayList<>();
        for (int i = 5*T; i < y.size(); i++) {
            Map<String, Double> result = model.decompose(y.get(i), i);
            decomposedTrend.add(result.get("trend"));
            decomposedSeasonal.add(result.get("seasonal"));
            decomposedTimestamp.add(i);
        }

        Plot plt = Plot.create();
//        plt.plot().add(timestamp, y);
        plt.plot().add(timestamp, trend);
        plt.plot().add(decomposedTimestamp, decomposedTrend);
        plt.plot().add(decomposedTimestamp, decomposedSeasonal);
//        plt.plot().add(timestamp, seasonal);
//        plt.plot().add(timestamp, residual);
        plt.show();
    }
}
