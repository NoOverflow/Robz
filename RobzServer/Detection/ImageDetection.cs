using Microsoft.ML;
using Robz.Structures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Threading.Tasks;

namespace Robz.Detection
{
    public class ImageDetection
    {
        private MLContext _Context = null;

        private void LogDetection(IEnumerable<IList<DetectionBoundingBox>> boundingBoxes)
        {
            foreach (var image in boundingBoxes)
                foreach (var box in boundingBoxes.ElementAt(0))
                    System.Diagnostics.Debug.Print(box.Label + " " + box.Confidence + " " + box.Dimension.X + " " + box.Dimension.Y + " " + box.Dimension.Width + " " + box.Dimension.Height + " " + box.Color);
        }

        public Image Detect(string ImageURL)
        {
            IEnumerable<ImageData> images = new List<ImageData>()
            {
                ImageData.FromURL(ImageURL)
            };
            IDataView imageDataView = _Context.Data.LoadFromEnumerable(images);
            ModelScorer modelScorer = new ModelScorer("Assets/Model/Model.onnx", _Context);
            IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);
            ModelParser parser = new ModelParser();
            var boundingBoxes = probabilities
                .Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));
            Image holder = Image.FromFile(images.ElementAt(0).ImagePath);

            LogDetection(boundingBoxes);
            return (TransformImage(Image.FromFile(images.ElementAt(0).ImagePath), boundingBoxes.ElementAt(0)));
        }

        private Image TransformImage(Image inputImage, IList<DetectionBoundingBox> boundingBoxes)
        {
            var originalImageHeight = inputImage.Height;
            var originalImageWidth = inputImage.Width;
            Image outputImage = (Image) inputImage.Clone();   
            Font drawFont = new Font("Calibri", 12, FontStyle.Bold);
            SolidBrush fontBrush = new SolidBrush(Color.Black);

            foreach (var box in boundingBoxes)
            {
                var x = (uint)Math.Max(box.Dimension.X, 0);
                var y = (uint)Math.Max(box.Dimension.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimension.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimension.Height);
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")} %)";
                Pen pen = new Pen(box.Color, 2f);
                SolidBrush colorBrush = new SolidBrush(box.Color);

                x = (uint)originalImageWidth * x / ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y /ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / ImageNetSettings.imageHeight;

                using (Graphics graph = Graphics.FromImage(outputImage))
                {
                    SizeF size = graph.MeasureString(text, drawFont);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    graph.CompositingQuality = CompositingQuality.HighQuality;
                    graph.SmoothingMode = SmoothingMode.HighQuality;
                    graph.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    graph.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    graph.DrawString(text, drawFont, fontBrush, atPoint);
                    graph.DrawRectangle(pen, x, y, width, height);
                }
            }

            return (outputImage);
        }

        public ImageDetection()
        {
            _Context = new MLContext();
        }
    }
}
