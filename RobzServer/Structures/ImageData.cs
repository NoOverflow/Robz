using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace Robz.Structures
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static ImageData FromPath(string path) => new ImageData()
        {
            ImagePath = path,
            Label = Path.GetFileNameWithoutExtension(path)
        };

        private static string SaveImage(string URL, ImageFormat format)
        {
            string temp = Path.GetTempFileName();

            using (WebClient client = new WebClient())
            {
                Stream stream = client.OpenRead(URL);
                Bitmap bitmap; bitmap = new Bitmap(stream);

                if (bitmap != null)
                    bitmap.Save(temp, format);
                stream.Flush();
                stream.Close();
            }
            return (temp);
        }

        public static ImageData FromURL(string URL) => new ImageData()
        {
            Label = "None",
            ImagePath = SaveImage(URL, ImageFormat.Png)
        };
    }
}
