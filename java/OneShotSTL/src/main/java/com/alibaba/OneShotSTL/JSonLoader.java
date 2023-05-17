package com.alibaba.OneShotSTL;

import java.util.HashMap;
import java.util.Map;
import java.nio.file.Paths;
import com.fasterxml.jackson.databind.ObjectMapper;

public class JSonLoader {
    public Map<String, Object> load(String inputFile) {
        Map<String, Object> data = new HashMap<>();
        try {
            ObjectMapper mapper = new ObjectMapper();
            Map<?, ?> map = mapper.readValue(Paths.get(inputFile).toFile(), Map.class);

            for (Map.Entry<?, ?> entry : map.entrySet()) {
                data.put(entry.getKey().toString(), entry.getValue());
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        return data;
    }
}
