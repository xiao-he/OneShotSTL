package com.alibaba.OneShotSTL;

import java.util.*;
import org.apache.log4j.Logger;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

/**
 * 2022/7/21
 *
 * @author xiao.hx@alibaba-inc.com
 */
public class OneShotSTLAlg {
    private static final Logger LOG = Logger.getLogger(OneShotSTLAlg.class);

    private int period = 0;
    private int shiftWindow = 0;
    private int maxIter = 5;
    private double orgLambda1 = 1.0;
    private double orgLambda2 = 0.5;
    private double orgLambda3 = 1.0;
    private double lambda1 = 1.0;
    private double lambda2 = 0.5;
    private double lambda3 = 1.0;
    private boolean isInitialized = false;
    private String f = "LS";
    private String h = "LAD";
    private String g = "LS";
    private double nSigmaThreshold = 5.0;
    private int nTrainPeriod = 8;

    // Used for Feature Engineering
    private int nPeriodBufferSize = 3;
//    private double[] trendBuffer;
//    private double[] seasonalBuffer;
//    private double[] residualBuffer;

    // Used for Online Decomposition
    private double[] lastTwoTrendBuffer;
    private double[] lastThreePeriodSeasonalBuffer;

    private int forecastStep = 0;
//    private int nPeriodForecastBufferLength = 0;
//    private Deque<Double> forecastBuffer;
//    private List<Double> decomposeTrendBuffer;
//    private double[] decomposeTrendBufferArray;
//    private NSigmaDetector forecastDetector;

    private double[][] A1;
    private double[][] A2;
    private double[][] A3;
    private double[][] A4;

    private DMatrixRMaj A1Matrix;
    private DMatrixRMaj A2Matrix;
    private DMatrixRMaj A3Matrix;
    private DMatrixRMaj A4Matrix;
    private DMatrixRMaj A1TMatrix;
    private DMatrixRMaj A2TMatrix;
    private DMatrixRMaj A3TMatrix;
    private DMatrixRMaj A1TA1Matrix;
    private DMatrixRMaj A4TMatrix;
    private DMatrixRMaj A3DiagMatrix;

    private DMatrixRMaj AIte0;

    private DMatrixRMaj A1LastRow;
    private DMatrixRMaj A2LastRow;
    private DMatrixRMaj A3LastRow;
    private DMatrixRMaj A4LastRow;

    private Deque<Double> b1;
    private Deque<Double> b2;
    private List<Deque<Double>> b3;
    private Deque<Double> b4;
    private double[] b1Array;
    private double[] b1Array2;
    private double[] b2Array;
    private List<double[]> b3Array;
    private double[] b4Array;

    private Map<Integer, Deque<Double>> w1;
    private Map<Integer, Deque<Double>> w2;
    private Map<Integer, List<Deque<Double>>> w3;
    private Map<Integer, Deque<Double>> w4;
    private Map<Integer, double[]> w1Array;
    private Map<Integer, double[]> w2Array;
    private Map<Integer, List<double[]>> w3Array;
    private Map<Integer, double[]> w4Array;

    private Map<Integer, OnlineBandedSystemAlg> onlineBandedSystemSolvers;

    private List<Integer> neighbors;

    private int t = 0;
    private int nSamples = 0;
    private CommonOps_DDRM ops;

    private NSigmaDetector trendDetector;
    private NSigmaDetector residDetector;
    private NSigmaDetector trendPlusResidDetector;

    private static int Band = 4;

    public OneShotSTLAlg(int period,
                         int shiftWindow,
                         int nSigmaWindow,
                         double lambda1,
                         double lambda2,
                         double lambda3,
                         int maxIter,
                         String f,
                         String h,
                         String g,
                         int ratio,
                         int forecastStep) {
        this(period, shiftWindow, nSigmaWindow, lambda1, lambda2, lambda3, maxIter, f, h, g, ratio);
//        nTrainPeriod = Math.max(8, forecastStep / period + 1);
//        nPeriodForecastBufferLength = Math.max(forecastStep / period + 1, 3);
//        this.forecastStep = forecastStep + 1;
        this.forecastStep = forecastStep;
    }

    public OneShotSTLAlg(int period,
                         int shiftWindow,
                         int nSigmaWindow,
                         double lambda1,
                         double lambda2,
                         double lambda3,
                         int maxIter,
                         String f,
                         String h,
                         String g,
                         int ratio) {
        assert period > 0;
        assert shiftWindow >= 0;
        assert nSigmaWindow == 0 || nSigmaWindow >= 1000;
        assert maxIter > 0;
        assert (f.equals("LS") | f.equals("LAD"));
        assert (h.equals("LS") | h.equals("LAD"));
        assert (g.equals("LS") | g.equals("LAD"));

        if (shiftWindow > 1.0 / ratio * period) {
            shiftWindow = period /ratio;
        }

        neighbors = new ArrayList<>();
        for (int i = 1; i < shiftWindow +1; i++) {
            neighbors.add(i);
            neighbors.add(-i);
        }

        this.period = period;
        this.shiftWindow = shiftWindow;
        this.maxIter = maxIter;
        this.orgLambda1 = lambda1;
        this.orgLambda2 = lambda2;
        this.orgLambda3 = lambda3;
        this.f = f;
        this.h = h;
        this.g = g;

        ops = new CommonOps_DDRM();
        trendDetector = new NSigmaDetector(nSigmaWindow);
        residDetector = new NSigmaDetector(nSigmaWindow);
        trendPlusResidDetector = new NSigmaDetector(nSigmaWindow);
//        forecastDetector = new NSigmaDetector(nSigmaWindow);

        reset();
    }

    private void reset() {
        t = 0;
        nSamples = 0;

        trendDetector.reset();
        residDetector.reset();
        trendPlusResidDetector.reset();

        lastTwoTrendBuffer = new double[2];
        lastThreePeriodSeasonalBuffer = new double[3 * period];

//        trendBuffer = new double[nPeriodBufferSize * period];
//        seasonalBuffer = new double[nPeriodBufferSize * period];
//        residualBuffer = new double[nPeriodBufferSize * period];

        b1 = new ArrayDeque<>();
        w1 = new HashMap<>();
        w1Array = new HashMap<>();
        for (int i = 0; i < maxIter; i++) {
            w1.put(i, new ArrayDeque<>());
        }

        b2 = new ArrayDeque<>();
        w2 = new HashMap<>();
        w2Array = new HashMap<>();
        for (int i = 0; i < maxIter; i++) {
            w2.put(i, new ArrayDeque<>());
        }

        b3 = new ArrayList<>();
        w3 = new HashMap<>();
        w3Array = new HashMap<>();
        b3.add(new ArrayDeque<>());
        b3.add(new ArrayDeque<>());
        for (int i = 0; i < maxIter; i++) {
            w3.put(i, new ArrayList<>());
            w3.get(i).add(new ArrayDeque<>());
            w3.get(i).add(new ArrayDeque<>());
        }

        b4 = new ArrayDeque<>();
        w4 = new HashMap<>();
        w4Array = new HashMap<>();
        for (int i = 0; i < maxIter; i++) {
            w4.put(i, new ArrayDeque<>());
        }

        onlineBandedSystemSolvers = new HashMap<>();

        double[] diagValue = new double[Band * 2];
        for (int j = 0; j < 4; j++) {
            diagValue[2 * j + 1] = 1;
        }
        A3DiagMatrix = ops.diag(diagValue);
    }

    public Map<String, double[]> initialize(List<Double> y) {
        if (y.size() / period <= 3) {
            period = period / 2;
        }
        int nPeriod = nTrainPeriod;
        List<Double> initialY;
        if (y.size() >= nPeriod * period) {
            initialY = y.subList(y.size() - nPeriod * period, y.size());
        } else {
            initialY = y;
        }
//        LOG.info(initialY.size());
        double[] trend;
        double[] seasonal;
        double[] resid;
        double[] initialYDouble = convertDouble(initialY.toArray());
        SeasonalTrendLoess.Builder builder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess smoother = builder.setPeriodic().setPeriodLength(period).setRobust().buildSmoother(initialYDouble);
        SeasonalTrendLoess.Decomposition stl = smoother.decompose();

        trend = stl.getTrend();
        seasonal = stl.getSeasonal();
        resid = stl.getResidual();

        isInitialized = true;

        double[] weights = {1, 10, 100, 1000, 10000, 100000};
//        double[] weights = {1};
        double bestWeight = 0.0;
        double minMAE = Double.MAX_VALUE;
        for (double weight : weights) {
            reset();
            lambda1 = orgLambda1 * weight;
            lambda3 = orgLambda3 * weight;
            Map<String, double[]> result = initialize(initialY, trend, seasonal, resid, nPeriod - 3);
            double mae = result.get("trendMAE")[0]
                    + result.get("residMAE")[0]
                    + result.get("seasonalMAE")[0];
//            LOG.info(weight+","+mae);
            if (mae < minMAE && Math.abs(mae - minMAE) > 1e-3) {
                minMAE = mae;
                bestWeight = weight;
            }
        }

        reset();
        lambda1 = orgLambda1 * bestWeight;
        lambda3 = orgLambda3 * bestWeight;
//        LOG.info(bestWeight);
        return initialize(initialY, trend, seasonal, resid, nPeriod - 3);
    }

    public Map<String, double[]> initialize(List<Double> y,
                                            double[] trend,
                                            double[] seasonal,
                                            double[] resid,
                                            int nPeriod) {
        if (trend.length / period - 3 < nPeriod) {
            nPeriod = trend.length / period - 3;
        }
        if (nPeriod < 0) {
            nPeriod = 0;
        }
        for (int i = 0; i < trend.length - nPeriod * period; i++) {
            try {
                trendDetector.add(trend[i]);
            } catch (Exception e) {
                LOG.info("");
            }
            residDetector.add(resid[i]);
            trendPlusResidDetector.add(trend[i] + resid[i]);
        }

        lastTwoTrendBuffer[0] = trend[trend.length - period * nPeriod - 2];
        lastTwoTrendBuffer[1] = trend[trend.length - period * nPeriod - 1];
        System.arraycopy(seasonal, seasonal.length - (nPeriod + 3) * period, lastThreePeriodSeasonalBuffer, 0, 3 * period);

//        System.arraycopy(trend, trend.length - (nPeriod + 3) * period, trendBuffer, 0, 3 * period);
//        System.arraycopy(seasonal, seasonal.length - (nPeriod + 3) * period, seasonalBuffer, 0, 3 * period);
//        System.arraycopy(resid, resid.length - (nPeriod + 3) * period, residualBuffer, 0, 3 * period);

        double[] trendNew = new double[period * nPeriod];
        double[] seasonalNew = new double[period * nPeriod];
        double[] residNew = new double[period * nPeriod];

        double trendMAE = 0.0;
        double seasonalMAE = 0.0;
        double residMAE = 0.0;
        int windowCopy = shiftWindow;
        shiftWindow = 0;
        for (int i = 0; i < period * nPeriod; i++) {
            Map<String, Double> result = decompose(y.get(y.size() - nPeriod * period + i), y.size() - nPeriod * period + i);
//            lastThreePeriodSeasonalBuffer[t] = lastThreePeriodSeasonalBuffer[t + period];
//            lastThreePeriodSeasonalBuffer[t + period] = lastThreePeriodSeasonalBuffer[t + 2 * period];
//            lastThreePeriodSeasonalBuffer[t + 2 * period] = result.get("seasonal");
//            lastTwoTrendBuffer[0] = lastTwoTrendBuffer[1];
//            lastTwoTrendBuffer[1] = result.get("trend");
            trendNew[i] = result.get("trend");
            seasonalNew[i] = result.get("seasonal");
//            trendBuffer = arrayDeleteAppend(trendBuffer, result.get("trend"));
//            seasonalBuffer = arrayDeleteAppend(seasonalBuffer, result.get("seasonal"));
//            residualBuffer = arrayDeleteAppend(residualBuffer, result.get("residual"));
            residNew[i] = y.get(y.size() - nPeriod * period + i) - trendNew[i] - seasonalNew[i];
            trendMAE += Math.abs(trend[y.size() - nPeriod * period + i] - result.get("trend"));
            seasonalMAE += Math.abs(seasonal[y.size() - nPeriod * period + i] - result.get("seasonal"));
            residMAE += Math.abs(resid[y.size() - nPeriod * period + i] - result.get("residual"));
        }
        shiftWindow = windowCopy;
        trendMAE /= period * nPeriod;
        seasonalMAE /= period * nPeriod;
        residMAE /= period * nPeriod;
        double[] trendMAEArray = {trendMAE};
        double[] seasonalMAEArray = {seasonalMAE};
        double[] residMAEArray = {residMAE};

        Map<String, double[]> result = new HashMap<>();
        result.put("trend", trend);
        result.put("seasonal", seasonal);
        result.put("resid", resid);
        result.put("trendNew", trendNew);
        result.put("seasonalNew", seasonalNew);
        result.put("residNew", residNew);
        result.put("trendMAE", trendMAEArray);
        result.put("seasonalMAE", seasonalMAEArray);
        result.put("residMAE", residMAEArray);

        return result;
    }

    public Map<String, double[]> predictAll(int i, int step) {
        t = i % period;
        double[] prediction = new double[step+1];
        double[] predTrend = new double[step+1];
        double[] predSeasonal = new double[step+1];
        for (int j = 0; j < step+1; j++) {
            int jj = j % period;
            double[] b3 = compute_b3ShiftValue(jj);
            predTrend[j] = (lastTwoTrendBuffer[0] + lastTwoTrendBuffer[1]) / 2;
            predSeasonal[j] = b3[1];
            prediction[j] = predTrend[j] + predSeasonal[j];
        }
        Map<String, double[]> result = new HashMap<>();
        result.put("prediction", prediction);
        result.put("predTrend", predTrend);
        result.put("predSeasonal", predSeasonal);
        return result;
    }

    public double[] predictAll(int i, int step, double yNew) {
        t = i % period;
//        double[] prediction = new double[step+1];
        double[] prediction = new double[step];
//        for (int j = 0; j < step+1; j++) {
        for (int j = 0; j < step; j++) {
            int jj = j % period;
            double[] b3 = compute_b3ShiftValue(jj);
//            if (jj < period / 2) {
//                double predTrend = forecast(trendBuffer, jj);
//                LOG.info(predTrend);
//                double predSeasonal = forecast(seasonalBuffer, jj);
//                prediction[j] = predSeasonal + predTrend;
//            } else {
                double predSeasonal = b3[1];
                double predTrend = (lastTwoTrendBuffer[0] + lastTwoTrendBuffer[1]) / 2;
                prediction[j] = predSeasonal + predTrend;
//            }
        }
        return prediction;
    }

    public Map<String, Double> predict(int i, int step, double yNew) {
        t = i % period;
        int j = step % period;
        double[] b3 = compute_b3ShiftValue(j);
        double prediction;
//        prediction = b3[1] + (lastTwoTrendBuffer[0] + lastTwoTrendBuffer[1]) / 2;
//        if (j < period / 2) {
//            double predTrend = forecast(trendBuffer, j);
//            double predSeasonal = forecast(seasonalBuffer, j);
//            prediction = predSeasonal + predTrend;
//        } else {
            double predSeasonal = b3[1];
            double predTrend = (lastTwoTrendBuffer[0] + lastTwoTrendBuffer[1]) / 2;
            prediction = predSeasonal + predTrend;
//        }
        Map<String, Double> result = new HashMap<>();
        result.put("prediction", prediction);
        return result;
    }

//    private double forecast(double[] buffer, int step) {
//        int p = 0;
//        int d = 1;
//        int q = 0;
//        int P = 0;
//        int D = 0;
//        int Q = 0;
//        int m = 0;
//        ForecastResult forecastResult = Arima.forecast_arima(buffer, step+1, new ArimaParams(p, d, q, P, D, Q, m));
//        double[] forecastData = forecastResult.getForecast();
//        return forecastData[step];
//    }

    public Map<String, Double> decompose(double yNew, int i) {
        t = i % period;
        nSamples += 1;
        Map<String, Double> result = decompose(yNew);
        double trend = result.get("trend");
        double seasonal = result.get("seasonal");
        double residual = result.get("residual");

        lastThreePeriodSeasonalBuffer[t] = lastThreePeriodSeasonalBuffer[t + period];
        lastThreePeriodSeasonalBuffer[t + period] = lastThreePeriodSeasonalBuffer[t + 2 * period];
        lastThreePeriodSeasonalBuffer[t + 2 * period] = seasonal;

        lastTwoTrendBuffer[0] = lastTwoTrendBuffer[1];
        lastTwoTrendBuffer[1] = trend;

//        trendBuffer = arrayDeleteAppend(trendBuffer, trend);
//        seasonalBuffer = arrayDeleteAppend(seasonalBuffer, seasonal);
//        residualBuffer = arrayDeleteAppend(residualBuffer, residual);

        double trendNSigma = trendDetector.detectAdd(trend);
        double residNSigma = residDetector.detectAdd(residual);
        double trendPlusResidNSigma = trendPlusResidDetector.detectAdd(trend + residual);

//            result.put("anomalyScore", Math.max(trendNSigma, residNSigma));
//            result.put("anomalyScore", trendPlusResidNSigma);
        result.put("anomalyScore", residNSigma);
//            result.put("anomalyScore", trendNSigma);
//            result.put("anomalyScore", Math.max(trendPlusResidNSigma, Math.max(trendNSigma, residNSigma)));

        return result;
    }

    private Map<String, Double> decompose(double yNew) {
        Map<String, Double> result = new HashMap<>();
        if (nSamples <= Band) {
            b1.add(yNew);
            if (nSamples == 1) {
                b2.add(2 * lastTwoTrendBuffer[0] - lastTwoTrendBuffer[1]);
                b4.add(lastTwoTrendBuffer[1]);
            } else if (nSamples == 2) {
                b2.add(-lastTwoTrendBuffer[1]);
                b4.add(0.0);
            } else {
                b2.add(0.0);
                b4.add(0.0);
            }
            compute_A1_A2_A3_A4();
            b1Array = convertDouble(b1.toArray());
            b2Array = convertDouble(b2.toArray());
            b4Array = convertDouble(b4.toArray());
            if (f.equals("LS")) {
                b1Array2 = new double[b1Array.length * 2];
                for (int i = 0; i < b1Array.length; i++){
                    b1Array2[i] = b1Array[i];
                    b1Array2[i + b1Array.length] = b1Array[i];
                }
            }
        } else {
            if (f.equals("LS")) {
                b1Array2 = arrayDeleteAppend(b1Array, yNew, yNew);
            }
            b1Array = arrayDeleteAppend(b1Array, yNew);
            b2Array = arrayDeleteAppend(b2Array, 0.0);
            b4Array = arrayDeleteAppend(b4Array, 0.0);
        }

        double[] x = decomposeAdaptive(yNew);
        double trend = x[x.length - 2];
        double seasonal = x[x.length - 1];
        double residual = yNew - trend - seasonal;
        result.put("trend", trend);
        result.put("seasonal", seasonal);
        result.put("residual", residual);
        return result;
    }

    private double[] decomposeAdaptive(double yNew) {
        if (shiftWindow == 0 || nSamples <= Band) {
            if (nSamples <= Band) {
                compute_b3(0);
                b3Array = new ArrayList<>();
                for (Deque<Double> b3i : b3) {
                    b3Array.add(convertDouble(b3i.toArray()));
                }
            } else {
                compute_b3Array(0, b3Array);
            }
            return onlineSolveProblem(w1Array, w2Array, w3Array,
                                      w4Array, b3Array, true);
        } else {
            Map<Integer, double[]> w1ArrayCopy = cloneMapArray(w1Array);
            Map<Integer, double[]> w2ArrayCopy = cloneMapArray(w2Array);
            Map<Integer, List<double[]>> w3ArrayCopy = cloneMapList(w3Array);
            Map<Integer, double[]> w4ArrayCopy = cloneMapArray(w4Array);
            List<double[]> b3ArrayCopy = cloneList(b3Array);
            compute_b3Array(0, b3ArrayCopy);
            double[] x = onlineSolveProblem(w1ArrayCopy, w2ArrayCopy, w3ArrayCopy,
                                            w4ArrayCopy, b3ArrayCopy, false);
            double residual = yNew - x[x.length - 1] - x[x.length - 2];
            double residNSigma = residDetector.detect(residual);
            if (residNSigma < nSigmaThreshold) {
                w1Array = w1ArrayCopy;
                w2Array = w2ArrayCopy;
                w3Array = w3ArrayCopy;
                w4Array = w4ArrayCopy;
                b3Array = b3ArrayCopy;
                for (int ite = 0; ite < maxIter; ite++) {
                    onlineBandedSystemSolvers.get(ite).update();
                }
                return x;
            } else {
                int minJ = 0;
                double minAbsResidual = Math.abs(residual);
                for (Integer j : neighbors) {
                    w1ArrayCopy = cloneMapArray(w1Array);
                    w2ArrayCopy = cloneMapArray(w2Array);
                    w3ArrayCopy = cloneMapList(w3Array);
                    w4ArrayCopy = cloneMapArray(w4Array);
                    b3ArrayCopy = cloneList(b3Array);
                    compute_b3Array(j, b3ArrayCopy);
                    x = onlineSolveProblem(w1ArrayCopy, w2ArrayCopy, w3ArrayCopy,
                                           w4ArrayCopy, b3ArrayCopy, false);
                    residual = yNew - x[x.length - 1] - x[x.length - 2];
                    if (minAbsResidual > Math.abs(residual)) {
                        minAbsResidual = Math.abs(residual);
                        minJ = j;
                    }
                }
                if (minJ == 0) {
                    return x;
                } else {
                    compute_b3Array(minJ, b3Array);
                    return onlineSolveProblem(w1Array, w2Array, w3Array,
                                              w4Array, b3Array, true);
                }
            }
        }
    }

    private double[] onlineSolveProblem(Map<Integer, double[]> w1Array,
                                        Map<Integer, double[]> w2Array,
                                        Map<Integer, List<double[]>> w3Array,
                                        Map<Integer, double[]> w4Array,
                                        List<double[]> b3Array,
                                        boolean updateModel) {
        double[] x = new double[2 * b1Array.length];
        for (int i = 0; i < b1Array.length; i++) {
            x[2*i] = b1Array[i];
        }
        for (int ite = 0; ite < maxIter; ite++) {
            if (ite == 0) {
                if (nSamples <= Band) {
                    w1.get(ite).add(1.0);
                    w2.get(ite).add(1.0);
                    for (int i = 0; i < w3.get(ite).size(); i++) {
                        w3.get(ite).get(i).add(1.0);
                    }
                    w4.get(ite).add(1.0);
                }
            } else {
                int index = b1Array.length - 1;
                if (nSamples <= Band) {
                    if (f.equals("LS")) {
                        w1.get(ite).add(1.0);
                    } else {
                        w1.get(ite).add(compute_w(x, A1LastRow, b1Array[index], f));
                    }
                    if (h.equals("LS")) {
                        w2.get(ite).add(1.0);
                    } else {
                        w2.get(ite).add(compute_w(x, A2LastRow, b2Array[index], h));
                    }
                    for (int i = 0; i < w3.get(ite).size(); i++) {
                        if (g.equals("LS")) {
                            w3.get(ite).get(i).add(1.0);
                        } else {
                            w3.get(ite).get(i).add(compute_w(x, A3LastRow, b3Array.get(i)[index], g));
                        }
                    }
                    if (h.equals("LS")) {
                        w4.get(ite).add(1.0);
                    } else {
                        w4.get(ite).add(compute_w(x, A4LastRow, b4Array[index], h));
                    }
                } else {
                    if (!f.equals("LS")) {
                        w1Array.put(ite, arrayDeleteAppend(w1Array.get(ite), compute_w(x, A1LastRow, b1Array[index], f)));
                    }
                    if (!h.equals("LS")) {
                        w2Array.put(ite, arrayDeleteAppend(w2Array.get(ite), compute_w(x, A2LastRow, b2Array[index], h)));
                    }
                    if (!g.equals("LS")) {
                        for (int i = 0; i < w3Array.get(ite).size(); i++) {
                            w3Array.get(ite).set(i, arrayDeleteAppend(w3Array.get(ite).get(i), compute_w(x, A3LastRow, b3Array.get(i)[index], g)));
                        }
                    }
                    if (!h.equals("LS")) {
                        w4Array.put(ite, arrayDeleteAppend(w4Array.get(ite), compute_w(x, A4LastRow, b4Array[index], h)));
                    }
                }
            }
            x  = onlineSolveLinearSystem(ite,
                                         w1Array,
                                         w2Array,
                                         w3Array,
                                         w4Array,
                                         b3Array,
                                         updateModel);
        }
        return x;
    }

    private double[] onlineSolveLinearSystem(int ite,
                                             Map<Integer, double[]> w1Array,
                                             Map<Integer, double[]> w2Array,
                                             Map<Integer, List<double[]>> w3Array,
                                             Map<Integer, double[]> w4Array,
                                             List<double[]> b3Array,
                                             boolean updateModel) {
        DMatrixRMaj A, b;
        if (nSamples <= Band) {
            w1Array.put(ite, convertDouble(w1.get(ite).toArray()));
            w2Array.put(ite, convertDouble(w2.get(ite).toArray()));
            List<double[]> w3ArrayList = new ArrayList<>();
            for (int i = 0; i < w3.get(ite).size(); i++) {
                w3ArrayList.add(convertDouble(w3.get(ite).get(i).toArray()));
            }
            w3Array.put(ite, w3ArrayList);
            w4Array.put(ite, convertDouble(w4.get(ite).toArray()));
            DMatrixRMaj[] Ab = compute_A_b(w1Array.get(ite), w2Array.get(ite), w3Array.get(ite), w4Array.get(ite));
            A = Ab[0];
            b = Ab[1];
            DMatrixRMaj x = new DMatrixRMaj(b.numRows, 1);
            ops.solve(A, b, x);
            if (nSamples == Band) {
                if (ite == 0) {
                    AIte0 = A;
                }
                if (updateModel) {
                    onlineBandedSystemSolvers.put(ite, new OnlineBandedSystemAlg(4));
                    onlineBandedSystemSolvers.get(ite).init(A, b);
                }
            }
            return x.getData();
        } else {
            if (ite == 0) {
                A = AIte0;
                b = compute_b(b3Array);
            } else {
                DMatrixRMaj[] Ab = compute_A_b_online(w1Array.get(ite),
                                                      w2Array.get(ite),
                                                      w3Array.get(ite),
                                                      w4Array.get(ite),
                                                      b3Array);
                A = Ab[0];
                b = Ab[1];
            }
            A = ops.extract(A, 2, A.numRows, 2, A.numCols);
            b = ops.extract(b, 2, b.numRows, 0, 1);
            return onlineBandedSystemSolvers.get(ite).onlineSolve(A, b, updateModel);
        }
    }

    private DMatrixRMaj[] compute_A_b(double[] w1Array,
                                      double[] w2Array,
                                      List<double[]> w3Array,
                                      double[] w4Array) {
        DMatrixRMaj A1TMatrixTmp = A1TMatrix.copy();
        ops.multCols(A1TMatrixTmp, w1Array);
        DMatrixRMaj A2TMatrixTmp = A2TMatrix.copy();
        ops.multCols(A2TMatrixTmp, w2Array);
        List<DMatrixRMaj> A3TMatrixTmp = new ArrayList<>();
        for (int i = 0; i < w3Array.size(); i++) {
            A3TMatrixTmp.add(A3TMatrix.copy());
            ops.multCols(A3TMatrixTmp.get(i), w3Array.get(i));
        }
        DMatrixRMaj A4TMatrixTmp = A4TMatrix.copy();
        ops.multCols(A4TMatrixTmp, w4Array);

        DMatrixRMaj A = new DMatrixRMaj(A1Matrix.numCols, A1Matrix.numCols);
        ops.mult(A1TMatrixTmp, A1Matrix, A);
        ops.multAdd(lambda1, A2TMatrixTmp, A2Matrix, A);
        for (int i = 0; i < A3TMatrixTmp.size(); i++) {
            ops.multAdd(lambda2, A3TMatrixTmp.get(i), A3Matrix, A);
        }
        ops.multAdd(lambda3, A4TMatrixTmp, A4Matrix, A);

        DMatrixRMaj b = new DMatrixRMaj(A1Matrix.numCols, 1);
        ops.mult(A1TMatrixTmp, new DMatrixRMaj(b1Array), b);
        ops.multAdd(lambda1, A2TMatrixTmp, new DMatrixRMaj(b2Array), b);
        for (int i = 0; i < A3TMatrixTmp.size(); i++) {
            ops.multAdd(lambda2, A3TMatrixTmp.get(i), new DMatrixRMaj(b3Array.get(i)), b);
        }
        ops.multAdd(lambda3, A4TMatrixTmp, new DMatrixRMaj(b4Array), b);

        DMatrixRMaj[] Ab = new DMatrixRMaj[2];
        Ab[0] = A;
        Ab[1] = b;

        return Ab;
    }

    private DMatrixRMaj[] compute_A_b_online(double[] w1Array,
                                             double[] w2Array,
                                             List<double[]> w3Array,
                                             double[] w4Array,
                                             List<double[]> b3Array) {
        DMatrixRMaj A1TMatrixTmp;
        if (f.equals("LS")) {
            A1TMatrixTmp = A1TMatrix;
        } else {
            A1TMatrixTmp = A1TMatrix.copy();
            ops.multCols(A1TMatrixTmp, w1Array);
        }
        DMatrixRMaj A2TMatrixTmp;
        if (h.equals("LS")) {
            A2TMatrixTmp = A2TMatrix;
        } else {
            A2TMatrixTmp = A2TMatrix.copy();
            ops.multCols(A2TMatrixTmp, w2Array);
        }
        List<DMatrixRMaj> A3TMatrixTmp = new ArrayList<>();
        if (!g.equals("LS")) {
            for (int i = 0; i < w3Array.size(); i++) {
                A3TMatrixTmp.add(A3TMatrix.copy());
                ops.multCols(A3TMatrixTmp.get(i), w3Array.get(i));
            }
        }
        DMatrixRMaj A4TMatrixTmp;
        if (h.equals("LS")) {
            A4TMatrixTmp = A4TMatrix;
        } else {
            A4TMatrixTmp = A4TMatrix.copy();
            ops.multCols(A4TMatrixTmp, w4Array);
        }

        DMatrixRMaj A = A1TA1Matrix.copy();
        if (!f.equals("LS")) {
            double[] w1ArrayNew = new double[Band * 2];
            for (int i = 0; i < w1Array.length; i++) {
                w1ArrayNew[2 * i] = w1Array[i];
                w1ArrayNew[2 * i + 1] = w1Array[i];
            }
            ops.multCols(A, w1ArrayNew);
        }
        ops.multAdd(lambda1, A2TMatrixTmp, A2Matrix, A);
        if (g.equals("LS")) {
            ops.addEquals(A, lambda2 * w3Array.size(), A3DiagMatrix);
        } else {
            for (int i = 0; i < w3Array.size(); i++) {
                double[] diagValue = new double[Band * 2];
                for (int j = 0; j < w3Array.get(i).length; j++) {
                    diagValue[2 * j + 1] = w3Array.get(i)[j];
                }
                ops.addEquals(A, lambda2, ops.diag(diagValue));
            }
        }
        ops.multAdd(lambda3, A4TMatrixTmp, A4Matrix, A);

        DMatrixRMaj b;
        if (f.equals("LS")) {
            b = new DMatrixRMaj(b1Array2);
        } else {
            b = new DMatrixRMaj(A1Matrix.numCols, 1);
            ops.mult(A1TMatrixTmp, new DMatrixRMaj(b1Array), b);
        }
        ops.multAdd(lambda1, A2TMatrixTmp, new DMatrixRMaj(b2Array), b);
        if (g.equals("LS")) {
            double[] b3Tmp = new double[b1Array2.length];
            for (double[] b3Arrayi : b3Array) {
                for (int i = 0; i < b3Arrayi.length; i++) {
                    b3Tmp[2 * i + 1] += b3Arrayi[i];
                }
            }
            ops.addEquals(b, lambda2, new DMatrixRMaj(b3Tmp));
        } else {
            for (int i = 0; i < A3TMatrixTmp.size(); i++) {
                ops.multAdd(lambda2, A3TMatrixTmp.get(i), new DMatrixRMaj(b3Array.get(i)), b);
            }
        }
        ops.multAdd(lambda3, A4TMatrixTmp, new DMatrixRMaj(b4Array), b);

        DMatrixRMaj[] Ab = new DMatrixRMaj[2];
        Ab[0] = A;
        Ab[1] = b;

        return Ab;
    }

    private DMatrixRMaj compute_b(List<double[]> b3Array) {
        DMatrixRMaj b;
        if (f.equals("LS")) {
            b = new DMatrixRMaj(b1Array2);
            double[] b3Tmp = new double[b1Array2.length];
            for (double[] b3Arrayi : b3Array) {
                for (int i = 0; i < b3Arrayi.length; i++) {
                    b3Tmp[2 * i + 1] += b3Arrayi[i];
                }
            }
            ops.addEquals(b, lambda2, new DMatrixRMaj(b3Tmp));
        } else {
            b = new DMatrixRMaj(A1Matrix.numCols, 1);
            ops.mult(A1TMatrix, new DMatrixRMaj(b1Array), b);
            for (double[] b3Arrayi : b3Array) {
                ops.multAdd(lambda2, A3TMatrix, new DMatrixRMaj(b3Arrayi), b);
            }
        }
        return b;
    }

    private double compute_w(double[] x, DMatrixRMaj A, double b, String func) {
        if (func.equals("LAD")) {
            double eta = 1e-5;
            double Ax = ops.dot(new DMatrixRMaj(x), A);
            return 0.5 / (Math.abs(Ax - b) + eta);
        }
        return 0.0;
    }

    private void compute_b3(int j) {
        double[] b3ShiftValue = compute_b3ShiftValue(j);
        b3.get(0).add(b3ShiftValue[0]);
        b3.get(1).add(b3ShiftValue[1]);
    }

    private void compute_b3Array(int j, List<double[]> b3Array) {
        double[] b3ShiftValue = compute_b3ShiftValue(j);
        b3Array.set(0, arrayDeleteAppend(b3Array.get(0), b3ShiftValue[0]));
        b3Array.set(1, arrayDeleteAppend(b3Array.get(1), b3ShiftValue[1]));
    }

    public double[] compute_b3ShiftValue(int j) {
        double[] b3ShiftValue = new double[2];
        if (j == 0) {
            b3ShiftValue[0] = lastThreePeriodSeasonalBuffer[t + period];
            b3ShiftValue[1] = lastThreePeriodSeasonalBuffer[t + 2 * period];
        } else if (j < 0) {
            if (t + j < 0) {
                b3ShiftValue[0] = lastThreePeriodSeasonalBuffer[t + period + j];
                b3ShiftValue[1] = lastThreePeriodSeasonalBuffer[t + 2 * period + j];
            } else {
                b3ShiftValue[0] = lastThreePeriodSeasonalBuffer[t + j];
                b3ShiftValue[1] = lastThreePeriodSeasonalBuffer[t + period + j];
            }
        } else {
            if (t + j < period) {
                b3ShiftValue[0] = lastThreePeriodSeasonalBuffer[t + period + j];
                b3ShiftValue[1] = lastThreePeriodSeasonalBuffer[t + 2 * period + j];
            } else {
                b3ShiftValue[0] = lastThreePeriodSeasonalBuffer[t + j];
                b3ShiftValue[1] = lastThreePeriodSeasonalBuffer[t + period + j];
            }
        }
        return b3ShiftValue;
    }

    private void compute_A1_A2_A3_A4() {
        if (nSamples == 1) {
            double[][] A1_ = {{1.0, 1.0}};
            double[][] A2_ = {{1.0, 0.0}};
            double[][] A3_ = {{0.0, 1.0}};
            double[][] A4_ = {{1.0, 0.0}};
            A1 = A1_;
            A2 = A2_;
            A3 = A3_;
            A4 = A4_;
            double[] A1LastRow_ = {1.0, 1.0};
            double[] A2LastRow_ = {1.0, 0.0};
            double[] A3LastRow_ = {0.0, 1.0};
            double[] A4LastRow_ = {1.0, 0.0};
            A1LastRow = new DMatrixRMaj(A1LastRow_);
            A2LastRow = new DMatrixRMaj(A2LastRow_);
            A3LastRow = new DMatrixRMaj(A3LastRow_);
            A4LastRow = new DMatrixRMaj(A4LastRow_);
        }

        if (nSamples == 2) {
            double[][] A1_ = {{1.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0, 1.0}};
            double[][] A2_ = {{1.0, 0.0, 0.0, 0.0},
                              {-2.0, 0.0, 1.0, 0.0}};
            double[][] A3_ = {{0.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 1.0}};
            double[][] A4_ = {{1.0, 0.0, 0.0, 0.0},
                              {-1.0, 0.0, 1.0, 0.0}};
            A1 = A1_;
            A2 = A2_;
            A3 = A3_;
            A4 = A4_;
            double[] A1LastRow_ = {0.0, 0.0, 1.0, 1.0};
            double[] A2LastRow_ = {-2.0, 0.0, 1.0, 0.0};
            double[] A3LastRow_ = {0.0, 0.0, 0.0, 1.0};
            double[] A4LastRow_ = {-1.0, 0.0, 1.0, 0.0};
            A1LastRow = new DMatrixRMaj(A1LastRow_);
            A2LastRow = new DMatrixRMaj(A2LastRow_);
            A3LastRow = new DMatrixRMaj(A3LastRow_);
            A4LastRow = new DMatrixRMaj(A4LastRow_);
        }

        if (nSamples == 3) {
            double[][] A1_ = {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 1.0, 1.0}};
            double[][] A2_ = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {-2.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                              {1.0, 0.0, -2.0, 0.0, 1.0, 0.0}};
            double[][] A3_ = {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
            double[][] A4_ = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, -1.0, 0.0, 1.0, 0.0}};
            A1 = A1_;
            A2 = A2_;
            A3 = A3_;
            A4 = A4_;
            double[] A1LastRow_ = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
            double[] A2LastRow_ = {1.0, 0.0, -2.0, 0.0, 1.0, 0.0};
            double[] A3LastRow_ = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
            double[] A4LastRow_ = {0.0, 0.0, -1.0, 0.0, 1.0, 0.0};
            A1LastRow = new DMatrixRMaj(A1LastRow_);
            A2LastRow = new DMatrixRMaj(A2LastRow_);
            A3LastRow = new DMatrixRMaj(A3LastRow_);
            A4LastRow = new DMatrixRMaj(A4LastRow_);
        }

        if (nSamples == 4) {
            double[][] A1_ = {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0}};
            double[][] A2_ = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {1.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0}};
            double[][] A3_ = {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
            double[][] A4_ = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0}};
            A1 = A1_;
            A2 = A2_;
            A3 = A3_;
            A4 = A4_;
            double[] A1LastRow_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
            double[] A2LastRow_ = {0.0, 0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0};
            double[] A3LastRow_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
            double[] A4LastRow_ = {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0};
            A1LastRow = new DMatrixRMaj(A1LastRow_);
            A2LastRow = new DMatrixRMaj(A2LastRow_);
            A3LastRow = new DMatrixRMaj(A3LastRow_);
            A4LastRow = new DMatrixRMaj(A4LastRow_);
        }

        A1Matrix = new DMatrixRMaj(A1);
        A2Matrix = new DMatrixRMaj(A2);
        A3Matrix = new DMatrixRMaj(A3);
        A4Matrix = new DMatrixRMaj(A4);
        A1TMatrix = new DMatrixRMaj(A1);
        A2TMatrix = new DMatrixRMaj(A2);
        A3TMatrix = new DMatrixRMaj(A3);
        A4TMatrix = new DMatrixRMaj(A4);
        ops.transpose(A1TMatrix);
        ops.transpose(A2TMatrix);
        ops.transpose(A3TMatrix);
        ops.transpose(A4TMatrix);

        A1TA1Matrix = new DMatrixRMaj();
        ops.mult(A1TMatrix, A1Matrix, A1TA1Matrix);
    }

    private double[] convertDouble(Object[] x) {
        double[] x_new = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            x_new[i] = (double) x[i];
        }
        return x_new;
    }

    private double[] arrayDeleteAppend(double [] a, double b) {
        double[] aNew = new double[a.length];
        System.arraycopy(a, 1, aNew, 0, a.length - 1);
        aNew[a.length-1] = b;
        return aNew;
    }

    private double[] arrayDeleteAppend(double [] a, double b, double c) {
        double[] aNew = new double[a.length * 2];
        System.arraycopy(a, 1, aNew, 0, a.length - 1);
        System.arraycopy(a, 1, aNew, a.length, a.length - 1);
        aNew[a.length-1] = b;
        aNew[aNew.length-1] = c;
        return aNew;
    }

    private Map<Integer, double[]> cloneMapArray(Map<Integer, double[]> w) {
        Map<Integer, double[]> wCopy = new HashMap<>();
        for (Integer i : w.keySet()) {
            double[] wi = new double[w.get(i).length];
            System.arraycopy(w.get(i), 0, wi, 0, wi.length);
            wCopy.put(i, wi);
        }
        return wCopy;
    }

    private List<double[]> cloneList(List<double[]> b) {
        List<double[]> bCopy = new ArrayList<>();
        for (double[] bi : b) {
            double[] biCopy = new double[bi.length];
            System.arraycopy(bi, 0, biCopy, 0, bi.length);
            bCopy.add(biCopy);
        }
        return bCopy;
    }

    private Map<Integer, List<double[]>> cloneMapList(Map<Integer, List<double[]>> w) {
        Map<Integer, List<double[]>> wCopy = new HashMap<>();
        for (Integer i : w.keySet()) {
            wCopy.put(i, new ArrayList<>());
            for (int j = 0; j < w.get(i).size(); j++) {
                double[] wij = new double[w.get(i).get(j).length];
                System.arraycopy(w.get(i).get(j), 0, wij, 0, wij.length);
                wCopy.get(i).add(wij);
            }
        }
        return wCopy;
    }

//    public Map<String, double[]> getBuffer() {
//        Map<String, double[]> result = new HashMap<>();
//        result.put("trend", trendBuffer);
//        result.put("seasonal", seasonalBuffer);
//        result.put("residual", residualBuffer);
//        return result;
//    }
}
