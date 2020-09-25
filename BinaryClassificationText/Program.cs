using System;
using System.IO;

namespace BinaryClassificationText
{
    class Program
    {
        static string GetTrainingData()
        {
            string datapath;
            do
            {
                Console.WriteLine("Enter a valid filename from /TrainingData for model training");
                string filename = Console.ReadLine().ToString();
                Console.WriteLine();
                datapath = Path.Combine(Environment.CurrentDirectory, "TrainingData", filename);
            }
            while (!File.Exists(datapath));

            return datapath;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("~~Binary Classification of Text~~\n");

            MachineLearning ml = new MachineLearning();
            ml.LoadTrainingData(GetTrainingData());

            Console.WriteLine("Creating and training the model");
            ml.Train();
            Console.WriteLine("Training complete\n");

            var metrics = ml.Evaluate();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}\n");

            ml.CreatePredictionEngine();

            string userinput = null;
            do
            {
                Console.WriteLine("Enter a string that is relative to the loaded training data ('exit' to close console)\n");
                userinput = Console.ReadLine().ToString();

                var formatinput = new TrainingData();
                formatinput.Text = userinput;

                var prediction = ml.Predict(formatinput);
                Console.WriteLine($"prediction: {prediction.Prediction}\n");
            }
            while (userinput != "exit");
        }
    }
}
