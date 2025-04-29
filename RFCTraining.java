import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.functions; 

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.io.IOException; 

public class RFCTraining {

    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().appName("Training").getOrCreate();

        // Define the schema for your CSV data
        List<StructField> fields = Arrays.asList(
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, true),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, true),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, true),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, true),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, true),
                DataTypes.createStructField("density", DataTypes.DoubleType, true),
                DataTypes.createStructField("pH", DataTypes.DoubleType, true),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, true),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, true),
                DataTypes.createStructField("quality", DataTypes.DoubleType, true)
        );
        StructType csvSchema = DataTypes.createStructType(fields);

        // Load and prepare the dataset
        Dataset<Row> wineDataset = spark.read()
                .format("csv")
                .schema(csvSchema)
                .option("header", true)
                .option("delimiter", ";")
                .option("quote", "\"")
                .option("ignoreLeadingWhiteSpace", true)
                .option("ignoreTrailingWhiteSpace", true)
                .load("file:///home/ec2-user/TrainingDataset.csv");

        // Remove quotes from column names 
        wineDataset = wineDataset.toDF(Stream.of(wineDataset.columns())
                .map(col -> col.replaceAll("\"", ""))
                .collect(Collectors.toList())
                .toArray(new String[0]));

        // Convert quality column to binary
        wineDataset = wineDataset.withColumn("quality",
                functions.when(wineDataset.col("quality").gt(7), 1.0).otherwise(0.0));


        // Create feature assembler
        String[] featureColumns = wineDataset.columns();
        featureColumns = Arrays.copyOf(featureColumns, featureColumns.length - 1);
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");
        wineDataset = vectorAssembler.transform(wineDataset);

        // Split into training and testing data
        Dataset<Row>[] splits = wineDataset.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testingData = splits[1];

        // Create and train SVM classifier
        RandomForestClassifier svmClassifier = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("features");
        RandomForestClassificationModel trainedSvmModel = svmClassifier.fit(trainingData);

        // Make predictions and evaluate
        Dataset<Row> testPredictions = trainedSvmModel.transform(testingData);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(testPredictions);
        System.out.println("F1 score: " + f1Score);

        // Save the model
        trainedSvmModel.write().overwrite().save("file:///home/ec2-user/trainingweights");
    }
}