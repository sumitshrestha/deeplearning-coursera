package test.dl.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import test.dl.service.LogisticRegressionService;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by Sumit Shrestha on 1/10/2019.
 */
@RestController
@RequestMapping("/nn")
public class HelloController {

    @Autowired
    private LogisticRegressionService logisticRegression;

    @RequestMapping("/")
    public String index() {
        return "Greetings from Spring Boot!";
    }

    @RequestMapping("/train")
    public Map train(@RequestParam final int modelType, @RequestParam int iteration, @RequestParam float rate) {
        switch (modelType) {
            case 1:
                return logisticRegression.train(iteration, rate);
            default:
                Map map = new LinkedHashMap<>();
                map.put("result", "invalid model type " + modelType);
                return map;
        }
    }

    @RequestMapping("/testFromTestSet")
    public Map test(@RequestParam final int modelType) {
        switch (modelType) {
            case 1:
                return logisticRegression.test();
            default:
                Map map = new LinkedHashMap<>();
                map.put("result", "invalid model type " + modelType);
                return map;
        }
    }

    @RequestMapping("/getParameters")
    public Map param(@RequestParam final String modelType) {
        switch (modelType) {
            case "1":
                Map parameters = logisticRegression.getParameters();
                System.out.println("this is parameters " + parameters.toString());
                return parameters;
            default:
                return new LinkedHashMap<>();
        }
    }
}