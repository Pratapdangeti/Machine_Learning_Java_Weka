package wekaprat;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.util.Random;


public class RegressionEnergyLoad {
	public static void main(String[] args) throws Exception {
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File("data/ENB2012_data.csv"));
		Instances data = loader.getDataSet();
//		System.out.println(data);
//		System.out.println(data.numInstances());
		
		data.setClassIndex(data.numAttributes()-2);
		Remove remove = new Remove();
		remove.setOptions(new String[] {"-R",data.numAttributes() + ""});
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new String[]{});
		System.out.println(eval.toSummaryString());
		
		M5P md5 = new M5P();
		md5.setOptions(new String[] {""});
		md5.buildClassifier(data);
		System.out.println(md5);
		
		eval.crossValidateModel(md5, data, 10, new Random(1), new String[]{});
		System.out.println(eval.toSummaryString());
		
		
		
		
	}
}
