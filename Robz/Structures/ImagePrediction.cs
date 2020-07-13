using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Robz.Structures
{
    public class ImagePrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
