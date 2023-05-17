package com.alibaba.OneShotSTL;

import org.apache.log4j.Logger;

import java.util.ArrayDeque;
import java.util.Deque;

public class NSigmaDetector {
    private static final Logger LOG = Logger.getLogger(NSigmaDetector.class);

    private int window;
    private int count = 0;
    private double sum = 0.0;
    private double sumSquare = 0.0;
    private Deque<Double> buffer;

    public NSigmaDetector(int window) {
        this.window = window;
        this.buffer = new ArrayDeque<>();
    }

    public void reset() {
        count = 0;
        sum = 0.0;
        sumSquare = 0.0;
        buffer = new ArrayDeque<>();
    }

    public int getBufferSize() {
        return buffer.size();
    }

    public double detectAdd(double value) {
        double nSigma = detect(value);
        add(value);
        return nSigma;
    }

    public void add(double value) {
        if (buffer.size() <= window) {
            if (window > 0) {
                buffer.add(value);
            }
            sum += value;
            sumSquare += value * value;
            count += 1;
        } else {
            double v = buffer.removeFirst();
            sum = sum - v + value;
            sumSquare = sumSquare - v * v + value * value;
            buffer.add(value);
        }
    }

    public double detect(double value) {
        int c = count;
        if (window > 0) {
            c = buffer.size();
        }
        double mean = sum / c;
        double std  = Math.sqrt(sumSquare / c - mean * mean);
        if (std < 1e-5) {
            return 0.0;
        } else {
            return Math.abs(value - mean) / std;
        }
    }

//    public double addGetMean(double value) {
//        add(value);
//        int c = count;
//        if (window > 0) {
//            c = buffer.size();
//        }
//        return sum / c;
//    }
}
