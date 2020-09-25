using Microsoft.ML;
using Microsoft.ML.Data;

namespace BinaryClassificationText
{
    /// <summary>
    /// Encapsulates ML.NET features
    /// </summary>
    public class MachineLearning
    {
        private MLContext mlContext;
        private IDataView dataViewTraining;
        private IDataView dataViewEvaluation;
        private ITransformer model;
        private PredictionEngine<TrainingData, PredictionData> predictionEngine;

        /// <summary>
        /// Class Constructor:
        /// Initialize the context for all ML.NET operations
        /// </summary>
        public MachineLearning()
        {
            mlContext = new MLContext();
        }

        /// <summary>
        /// Loads a dataset from a file.  
        /// Splits the input dataset into training data (90%) and evaluation data (10%)
        /// </summary>
        /// <param name="datapath"></param>
        public void LoadTrainingData(string datapath)
        {
            //convert training data into IDataView
            var dataView = mlContext.Data.LoadFromTextFile<TrainingData>(datapath, hasHeader: false);

            //split training data into two IDataViews (90% training/10% evaluation)
            var splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            dataViewTraining = splitDataView.TrainSet;
            dataViewEvaluation = splitDataView.TestSet;
        }

        /// <summary>
        /// Creates the pipeline and defines workflows within:
        ///     1. Transforms text column column to a featurized vector of floats
        ///     2. Defines the binary classification training algorithm, Fast Tree
        ///     3. Trains fast tree algorithm and creates the model using the training data
        /// </summary>
        public void Train()
        {
            //transform text and define algorithm
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(TrainingData.Text))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            //train the algorithm and create model
            model = pipeline.Fit(dataViewTraining);
        }

        /// <summary>
        /// Evaluates the model's accuracy with evaluation data 
        /// </summary>
        /// <returns>BinaryClassificationMetrics to report accuracy to console</returns>
        public BinaryClassificationMetrics Evaluate()
        {
            //transform evaluation data for the model
            var predictions = model.Transform(dataViewEvaluation);

            //evaluate prediction on the basis of "Label"
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            return metrics;
        }

        /// <summary>
        /// Creates prediction engine for prediction for a one-time prediction
        /// </summary>
        public void CreatePredictionEngine()
        {
            predictionEngine = mlContext.Model.CreatePredictionEngine<TrainingData, PredictionData>(model);
        }

        /// <summary>
        /// Uses the class's prediction engine to make a prediction on one-time instance of data
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public PredictionData Predict(TrainingData input)
        {
            var prediction = predictionEngine.Predict(input);

            return prediction;
        }
    }
}
