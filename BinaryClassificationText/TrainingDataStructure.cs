using Microsoft.ML.Data;

namespace BinaryClassificationText
{
    /// <summary>
    /// Defines the structure of the training data
    /// </summary>
    public class TrainingData
    {
        /// <summary>
        /// The text to be classified
        /// </summary>
        [LoadColumn(0)]
        public string Text { get; set; }

        /// <summary>
        /// The boolean value for the binary classification
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public bool Classification { get; set; }
    }

    /// <summary>
    /// Defines the structure of the prediction
    /// </summary>
    public class PredictionData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
