package com.alibaba.OneShotSTL;

import org.junit.Test;

public class testJSonLoader {
    @Test
    public void testDataLoader() {
        String inputFile = "/Users/xiaohe/Projects/OneShotSTL-supp/data/syn1.json";
        JSonLoader loader = new JSonLoader();
        loader.load(inputFile);
    }
}
