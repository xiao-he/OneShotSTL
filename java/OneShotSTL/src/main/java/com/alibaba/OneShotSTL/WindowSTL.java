package com.alibaba.OneShotSTL;

import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WindowSTL {
    private int period;
    private int slidingWindow;
    private double[] slidingWindowY;
    public WindowSTL(int period, int slidingWindow) {
        this.period = period;
        this.slidingWindow = slidingWindow;
    }

    public void initialize(List<Double> y) {
        if (y.size() > slidingWindow) {
            y = y.subList(y.size() - slidingWindow, y.size());
        }
        slidingWindowY = convertDouble(y.toArray());
    }

    public Map<String, Double> decompose(double yNew) {
        double[] slidingWindowYNew = new double[slidingWindow];
        System.arraycopy(slidingWindowY, 1, slidingWindowYNew, 0, slidingWindow - 1);
        slidingWindowYNew[slidingWindow - 1] = yNew;
        slidingWindowY = slidingWindowYNew;
        Map<String, Double> result = new HashMap<>();
        SeasonalTrendLoess.Builder builder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess smoother = builder.setPeriodic().setPeriodLength(period).setRobust().buildSmoother(slidingWindowY);
        SeasonalTrendLoess.Decomposition stl = smoother.decompose();
        result.put("trend", stl.getTrend()[slidingWindow - 1]);
        result.put("seasonal", stl.getSeasonal()[slidingWindow - 1]);
        result.put("residual", stl.getResidual()[slidingWindow - 1]);
        return result;
    }

    private double[] convertDouble(Object[] x) {
        double[] x_new = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            x_new[i] = (double) x[i];
        }
        return x_new;
    }
}
