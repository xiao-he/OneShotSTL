package com.alibaba.OneShotSTL;

import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Exp {
    private static final Logger LOG = Logger.getLogger(Exp.class);

    private static void printUsage(Options options) {
        HelpFormatter hlpfrmt = new HelpFormatter();
        hlpfrmt.printHelp("OneShotSTL", options);
    }

    public static void main(String[] args) throws Exception {
        CommandLineParser parser = new PosixParser();
        Options options = new Options();
        options.addOption(null, "task", true, "Task: decompose, anomaly or forecast");
        options.addOption(null, "method", true, "Method: OneShotSTL or WindowSTL");
        options.addOption(null, "in", true, "input json file");
        options.addOption(null, "out", true, "output json file");
        options.addOption(null, "period", true, "user specified input period");
        options.addOption(null, "slidingWindow", true, "sliding window size for WindowSTL");
        options.addOption(null, "shiftWindow", true, "shift window size for OneShotSTL");
        options.addOption(null, "onlyInit", false, "only perform init");
        options.addOption(null, "maxOnlineDecomposeNum", true, "the maximum number of points for online decomposition");
        options.addOption(null, "predStep", true, "prediction step for forecast task");
        CommandLine argsLine = null;
        try {
            argsLine = parser.parse(options, args);
        } catch (UnrecognizedOptionException | MissingArgumentException e) {
            LOG.error(e.getMessage());
            printUsage(options);
            System.exit(1);
        }

        String inputFileName = argsLine.getOptionValue("in");
        try {
            JSonLoader loader = new JSonLoader();
            Map<String, Object> data = loader.load(inputFileName);
            int T = (int) data.get("period");
            if (argsLine.hasOption("period")) {
                T = Integer.parseInt(argsLine.getOptionValue("period"));
            }
            List<Double> y = (List<Double>) data.get("ts");
            String method = argsLine.getOptionValue("method");
            if (!method.equalsIgnoreCase("OneShotSTL") && !method.equalsIgnoreCase("WindowSTL")) {
                LOG.error("Unknown method: " + method);
                printUsage(options);
                System.exit(1);
            }
            int window = 0;
            if (method.equalsIgnoreCase("OneShotSTL")) {
                window = Integer.parseInt(argsLine.getOptionValue("shiftWindow"));
            } else {
                window = Integer.parseInt(argsLine.getOptionValue("slidingWindow"));
            }
            String task = argsLine.getOptionValue("task");
            int trainTestSplit = (int) data.get("trainTestSplit");
            Map<String, Object> result = new HashMap<>();
            if (task.equalsIgnoreCase("decompose")) {
                if (argsLine.hasOption("period")) {
                    trainTestSplit = 5 * T;
                }
                OneShotSTLAlg oneShotSTLModel = new OneShotSTLAlg(T, window, 10000, 1.0, 0.5, 1.0, 8, "LS", "LAD", "LS", 1);
                WindowSTL windowSTLModel = new WindowSTL(T, window);
                List<Double> initY = y.subList(0, trainTestSplit);
                List<Double> trend = new ArrayList<>();
                List<Double> seasonal = new ArrayList<>();
                List<Double> residual = new ArrayList<>();
                if (method.equalsIgnoreCase("OneShotSTL")) {
                    oneShotSTLModel.initialize(initY);
                } else {
                    windowSTLModel.initialize(initY);
                }
                if (!argsLine.hasOption("onlyInit")) {
//                    long begin = System.nanoTime();
                    int maxOnlineDecomposeNum = y.size();
                    if (argsLine.hasOption("maxOnlineDecomposeNum")) {
                        maxOnlineDecomposeNum = trainTestSplit + Integer.parseInt(argsLine.getOptionValue("maxOnlineDecomposeNum"));
                    }
                    for (int i = trainTestSplit; i < maxOnlineDecomposeNum; i++) {
                        Map<String, Double> res;
                        if (method.equalsIgnoreCase("OneShotSTL")) {
                            res = oneShotSTLModel.decompose(y.get(i), i - trainTestSplit);
                        } else {
                            res = windowSTLModel.decompose(y.get(i));
                        }
                        trend.add(res.get("trend"));
                        seasonal.add(res.get("seasonal"));
                        residual.add(res.get("residual"));
                    }
//                    long end = System.nanoTime();
//                    double rt = (1.0 * (end - begin) / 1000000000);
//                    LOG.info(T+","+trainTestSplit);
//                    LOG.info(1.0 * rt / (y.size() - trainTestSplit));
                }
                result.put("trend", trend);
                result.put("seasonal", seasonal);
                result.put("residual", residual);
            } else if (task.equalsIgnoreCase("anomaly")) {
//                T = Math.min(T, trainTestSplit / 4);
                OneShotSTLAlg oneShotSTLModel = new OneShotSTLAlg(T, window, 10000, 1.0, 0.5, 1.0, 8, "LS", "LAD", "LS", 1);
                List<Double> initY = y.subList(0, trainTestSplit);
                oneShotSTLModel.initialize(initY);
                List<Double> anomlyScores = new ArrayList<>();
                for (int i = trainTestSplit; i < y.size(); i++) {
                    Map<String, Double> res = oneShotSTLModel.decompose(y.get(i), i - trainTestSplit);
                    anomlyScores.add(res.get("anomalyScore"));
                }
                result.put("anomalyScore", anomlyScores);
            } else if (task.equalsIgnoreCase("forecast")) {
                int step = Integer.parseInt(argsLine.getOptionValue("predStep"));
                OneShotSTLAlg oneShotSTLModel = new OneShotSTLAlg(T, window, 10000, 1.0, 0.5, 1.0, 8, "LS", "LAD", "LS", 1);
                List<Double> initY = y.subList(0, trainTestSplit);
                oneShotSTLModel.initialize(initY);
                List<double[]> allPred = new ArrayList<>();
                List<double[]> allReal = new ArrayList<>();
                double mae = 0.0;
                int count = 0;
                for (int i = trainTestSplit; i < y.size(); i++) {
                    if (i + step < y.size()) {
                        double[] prediction = oneShotSTLModel.predictAll(i - trainTestSplit, step, y.get(i-1));
                        allPred.add(prediction);
                        oneShotSTLModel.decompose(y.get(i), i - trainTestSplit);
                        double[] realval = new double[step];
                        for (int j = 0; j < step; j++) {
                            realval[j] = y.get(i+j);
                            mae += Math.abs(prediction[j] - realval[j]);
                            count += 1;
                        }
                        allReal.add(realval);
                    }
                }
                result.put("predStep", step);
                result.put("pred", allPred);
                result.put("real", allReal);
//                LOG.info(mae / count);
            } else {
                LOG.error("Unknown task: " + task);
                printUsage(options);
                System.exit(1);
            }

            if (argsLine.hasOption("out")) {
                ObjectMapper mapper = new ObjectMapper();
                String outputFileName = argsLine.getOptionValue("out");
                int lastIndex = outputFileName.lastIndexOf("/");
                String folder = outputFileName.substring(0, lastIndex);
                if (!folder.equalsIgnoreCase("")) {
                    Files.createDirectories(Paths.get(folder));
                }
                mapper.writeValue(Paths.get(outputFileName).toFile(), result);
            }
        } catch (Exception e) {
            LOG.info(inputFileName);
            e.printStackTrace();
            printUsage(options);
            System.exit(1);
        }
    }
}
