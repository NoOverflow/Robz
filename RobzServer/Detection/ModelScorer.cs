using Microsoft.ML;
using Microsoft.ML.Data;
using Robz.Structures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;

namespace Robz.Detection
{
    public struct TinyYoloModelSettings
    {
        public const string ModelInput = "image";

        public const string ModelOutput = "grid";
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }

    public class ModelScorer
    {
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        private IList<DetectionBoundingBox> _boundingBoxes = new List<DetectionBoundingBox>();

        public ModelScorer(string modelLocation, MLContext mlContext)
        {
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        private ITransformer LoadModel(string modelPath)
        {
            IDataView emptyView = mlContext.Data.LoadFromEnumerable(new List<ImageData>());
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

            return (pipeline.Fit(emptyView));
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            IDataView scoredData = model.Transform(testData);
            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

            return (probabilities);
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);

            return (PredictDataUsingModel(data, model));
        }
    }
}
