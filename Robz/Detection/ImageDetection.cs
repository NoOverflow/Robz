using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Robz.Detection
{
    public class ImageDetection
    {
        private MLContext _Context = null;

        public void Detect()
        {

        }

        public ImageDetection()
        {
            _Context = new MLContext();
        }
    }
}
