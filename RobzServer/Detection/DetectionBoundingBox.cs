using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;

namespace Robz.Detection
{
    public class DetectionBoundingBox
    {
        public Dimension Dimension { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
        public Color Color { get; set; }

        public RectangleF Rect
        {
            get
            {
                return new RectangleF(Dimension.X, Dimension.Y, Dimension.Width, Dimension.Height);
            }
        }
    }
}
