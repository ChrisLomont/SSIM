/*
MIT LICENSE

Copyright (c) 2009-2023 Chris Lomont

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

// Compute Structural Similarity Index (SSIM) image quality metrics                                                                                 
// See http://www.ece.uwaterloo.ca/~z70wang/research/ssim/                                                                                          
// Also computes PSNR, MSE, RMSE image quality metrics
// C++ and C# implementations Copyright Chris Lomont 2009-2023                                                                                        
// based on my old C# code at https://lomont.org/software/misc/ssim/SSIM.html
// Send fixes and comments to WWW.LOMONT.ORG                                                                                                        

/* History:
 * June 2023
 * 0.93 - Fix to ensure matches original on common datasets  
 *        TODO - see (provide link to blog post, github)
 *
 * Jan 2019
 * 0.92 - Made C++ version in parallel
 *      - Removed dependency on non dotnet CORE things
 *      - Made API static to make easier to use
 *      - added MSE, RMSE, PSNR to add more simple metrics
 *      - renamed to ImageMetrics.cs
 *
 * June 2011                                                                                                                                        
 * 0.91 - Fix to Bitmap creation to prevent locking file handles                                                                                    
 *                                                                                                                                                  
 * Sept 2009                                                                                                                                        
 * 0.9  - Initial Release                                                                                                                           
 *                                                                                                                                                  
 */



namespace Lomont.Graphics;

public static class ImageMetrics
{
    /// <summary>                                                                                                                               
    /// version of the ImageMetrics code                                                                                                            
    /// </summary>                                                                                                                              
    public static string Version => "0.93";

    /// <summary>
    /// get grayscale pixel in 0-1
    /// </summary>
    public delegate double GetPixel (int i, int j);

    // Mean Squared Error
    public static double Mse(int width, int height, GetPixel getPixel1, GetPixel getPixel2)
    {
        return ComputeMse(width, height, getPixel1, getPixel2);
    }

    // Root Mean Squared Error
    public static double Rmse(int width, int height, GetPixel getPixel1, GetPixel getPixel2)
    {
        return Math.Sqrt(Mse(width, height, getPixel1, getPixel2));
    }

    // compute Peak Signal-to-Noise Ratio
    public static double Psnr(int width, int height, GetPixel getPixel1, GetPixel getPixel2)
    {
        // 3 channel PSNR idea https://dsp.stackexchange.com/questions/71845/defining-the-snr-or-psnr-for-color-images-3-channel-rgb-files
        // see also https://groups.google.com/g/sci.image.processing/c/0iypIGoJf7g
        double mse = Mse(width, height, getPixel1, getPixel2);
        // todo - mse ~ 0 case
        return 10.0 * Math.Log10(1.0 / mse);
    }


    // compute SSIM from greyscale single plane image, colors in range 0-1

    // compute SSIM on one channel
    // expects grayscale, linear color space
    public static double Ssim(
        int width, int height, 
        GetPixel getPixel1, GetPixel getPixel2,
        double L = 1.0,   // color depth - 255 for bytewise
        // constants from the paper, with default values
        double K1 = 0.01,
        double K2 = 0.03
    )
    {
        return ComputeSsim(width, height, getPixel1, getPixel2, L, K1, K2);
    }

    // mimic MATLAB rgb2gray https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    // note this uses a weird convention of 0.2989 for the coefficient of red instead
    // of the coefficient 0.299
    public static double Rgb2Gray(double r, double g, double b)
    {
        return  0.2989 * r + 0.5870 * g + 0.1140 * b;
    }

    #region Implementation


    // Mean Squared Error
    static double ComputeMse(int width, int height, GetPixel getPixel1, GetPixel getPixel2)
    {
        double sum = 0.0;
        for (int j = 0; j<height; ++j)
        for (int i = 0; i<width; ++i)
        {
            var v1 = getPixel1(i, j);
            var v2 = getPixel2(i, j);
            var del = v1 - v2;
            sum += del* del;
            if (Double.IsNaN(sum))
                return sum; // early fails
        }
        return sum / (width* height);
    }

#if false // todo - remove
        /// <summary>                                                                                                                               
        /// Compute from two linear ushort grayscale images with given size and bitdepth                                                            
        /// </summary>                                                                                                                              
        /// <param name="img1">Image 1 data</param>                                                                                                 
        /// <param name="img2">Image 2 data</param>                                                                                                 
        /// <param name="w">width</param>                                                                                                           
        /// <param name="h">height</param>                                                                                                          
        /// <param name="depth">Bit depth (1-16)</param>                                                                                            
        /// <returns></returns>                                                                                                                     
        internal double Index(ushort[] img1, ushort[] img2, int w, int h, int depth)
        {
            L = (1 << depth) - 1;
            return ComputeSSIM(ConvertLinear(img1, w, h), ConvertLinear(img2, w, h));
        }

        /// <summary>                                                                                                                               
        /// Take two System.Drawing.Bitmaps                                                                                                         
        /// </summary>                                                                                                                              
        /// <param name="img1"></param>                                                                                                             
        /// <param name="img2"></param>                                                                                                             
        /// <returns></returns>                                                                                                                     
        internal double Index(Bitmap img1, Bitmap img2)
        {
            L = 255; // todo - this assumes 8 bit, but color conversion later is always 8 bit, so ok?                                               
            return ComputeSSIM(ConvertBitmap(img1), ConvertBitmap(img2));
        }

        /// <summary>                                                                                                                               
        /// Take two filenames                                                                                                                      
        /// </summary>                                                                                                                              
        /// <param name="filename1"></param>                                                                                                        
        /// <param name="filename2"></param>                                                                                                        
        /// <returns></returns>                                                                                                                     
        internal double Index(string filename1, string filename2)
        {
            using (var b1 = new Bitmap(filename1))
            using (var b2 = new Bitmap(filename2))
                return Index(b1, b2);
        }
#endif


    #region Locals                                                                                                                              
    // default settings, names from paper                                                                                                       
    //internal double K1 = 0.01, K2 = 0.03;
    //internal double L = 255;
    //readonly Array2D window = Gaussian(11, 1.5);
    #endregion

    // compute SSIM on one channel
    // expects grayscale, linear color space
    static double ComputeSsim(
        int width, int height, 
        GetPixel getPixel1, GetPixel getPixel2,
        // constants from the paper, with default values
        double L = 1.0,   // color depth - 255 for bytewise?
        double K1 = 0.01,
        double K2 = 0.03
    )
    {
        Array2D window = Gaussian(11, 1.5);
        Array2D img1 = new(width, height, getPixel1);
        Array2D img2 = new(width, height, getPixel2);

        // uses notation from paper                                                                                                             
        // automatic downsampling                                                                                                               
        int f = (int)Math.Max(1, Math.Round(Math.Min(width, height) / 256.0));
        if (f > 1)
        {         
            // simple low-pass filter, subsamples by f
            img1 = SubSample(img1, f);
            img2 = SubSample(img2, f);
        }

        // normalize window - todo - do in window set {}                                                                                        
        double scale = 1.0 / window.Total;
        Array2D.Op((i, j) => window[i, j] * scale, window);

        // image statistics
        var mu1 = Filter(img1, window);
        var mu2 = Filter(img2, window);
        var mu1mu2 = mu1 * mu2;
        var mu1Sq = mu1 * mu1;
        var mu2Sq = mu2 * mu2;
        var sigma12 = Filter(img1 * img2, window) - mu1mu2;
        var sigma1Sq = Filter(img1 * img1, window) - mu1Sq;
        var sigma2Sq = Filter(img2 * img2, window) - mu2Sq;

        double C1 = K1 * L; C1 *= C1;
        double C2 = K2 * L; C2 *= C2;

        var ssimMap = (2 * mu1mu2 + C1) * (2 * sigma12 + C2) /
                       ((mu1Sq + mu2Sq + C1) * (sigma1Sq + sigma2Sq + C2));

        // average all values
        return ssimMap.Total / (ssimMap.Width * ssimMap.Height);
    } // ComputeSSIM                                                                                                                    


    #region Array2D                                                                                                                                
    /// <summary>                                                                                                                               
    /// Hold a array2d of doubles as an array with appropriate operators                                                                           
    /// </summary>                                                                                                                              
    class Array2D
    {
        readonly double[,] data;
        internal readonly int Width;
        internal readonly int Height;

        public Array2D(int w, int h)
        {
            data = new double[w, h];
            Width = w;
            Height = h;
        }
        // create array2d from pixel source
        public Array2D(int w, int h, GetPixel pixels)
        {
            data = new double[w, h];
            Width = w;
            Height = h;
            for (var j = 0; j < Height; ++j)
            for (var i = 0; i < Width; ++i)
            {
                this[i, j] = pixels(i, j);
            }
        }

        // Indexer for the i,j item                                                                                                        
        internal double this[int i, int j]
        {
            get => data[i, j];
            set => data[i, j] = value;
        }

        // sum of all values in array2d
        internal double Total
        {
            get
            {
                double s = 0;
                foreach (var d in data) s += d;
                return s;
            }
        }

        // componentwise addition of array2d                                                                                                     
        public static Array2D operator +(Array2D a, Array2D b)
        {
            return Op((i, j) => a[i, j] + b[i, j], new Array2D(a.Width, a.Height));
        }

        // componentwise subtraction of array2d                                                                                                  
        public static Array2D operator -(Array2D a, Array2D b)
        {
            return Op((i, j) => a[i, j] - b[i, j], new Array2D(a.Width, a.Height));
        }

        // componentwise multiplication by constant
        public static Array2D operator *(double val, Array2D b)
        {
            return Op((i, j) => val * b[i, j], new Array2D(b.Width, b.Height));
        }
        // componentwise addition of constant
        public static Array2D operator +(Array2D b, double val)
        {
            return Op((i, j) => val + b[i, j], new Array2D(b.Width, b.Height));
        }

        // componentwise multiplication of array2d                                                                                               
        public static Array2D operator *(Array2D a, Array2D b)
        {
            return Op((i, j) => a[i, j] * b[i, j], new Array2D(a.Width, a.Height));
        }

        // componentwise division of array2d                                                                                                     
        public static Array2D operator /(Array2D a, Array2D b)
        {
            return Op((i, j) => a[i, j] / b[i, j], new Array2D(a.Width, a.Height));
        }

        /// <summary>                                                                                                                           
        /// Generic function maps (i,j) onto the given array2d                                                                                     
        /// </summary>                                                                                                                          
        public static Array2D Op(Func<int, int, double> f, Array2D g)
        {
            int w = g.Width, h = g.Height;
            for (int i = 0; i < w; ++i)
            for (int j = 0; j < h; ++j)
                g[i, j] = f(i, j);
            return g;
        }

    } // class Array2d                                                                                                                         
    #endregion //Array2d                                                                                                                           

    // Apply filter to signal, return only center part.
    // filter should be odd sized
    static Array2D Filter(Array2D signal, Array2D filter)
    {
        int signalW = signal.Width, signalH = signal.Height;
        int filterW = filter.Width, filterH = filter.Height;

        // dest item size:
        int resultW = signalW - filterW + 1, resultH = signalH - filterH + 1;
        Array2D c = new(resultW, resultH);

        // loop over dest samples
        for (var j = 0; j<resultH; ++j)
        for (var i = 0; i<resultW; ++i)
        {
            double sum = 0;

            // loop over filter
            for (var fj = 0; fj<filterH; ++fj)
            for (var fi = 0; fi<filterW; ++fi)
            {
                // signal coords:
                int si = i + fi;
                int sj = j + fj;

                // convolve
                sum += signal[si, sj]* filter[fi, fj];
            }

            c[i, j] = sum;
        }

        return c;
    }

    /// <summary>                                                                                                                               
    /// Create a gaussian window of the given size and standard deviation                                                                       
    /// </summary>                                                                                                                              
    /// <param name="size">Odd number</param>                                                                                                   
    /// <param name="sigma">Gaussian std deviation</param>                                                                                      
    /// <returns></returns>                                                                                                                     
    static Array2D Gaussian(int size, double sigma)
    {
        Array2D filter = new(size, size);
        double s2 = 2 * sigma * sigma;
        int c = size / 2;
        filter = Array2D.Op(
            (i,j)=>
                
            {
                double dx = i - c;
                double dy = j - c;
                return Math.Exp((-(dx * dx + dy * dy) / s2));
            }, filter);

        // Note original SSIM paper does not normalize the filter
        // But to get same results it is required

        return (1.0 / filter.Total) * filter;
    }

    // reflect signal in [0,max)
    // assumes not too far off edge
    static int Reflect(int val, int max)
    {
        if (val < 0) val = (-val) - 1;
        if (val >= max) val = 2 * max - val - 1;
        return val;
    }

    /// <summary>                                                                                                                               
    /// subsample a grid by step size, averaging each box into the result value                                                                 
    /// </summary>                                                                                                                              
    /// <returns></returns>                                                                                                                     
    static Array2D SubSample(Array2D img, int size)
    {
        int ow = img.Width;
        int oh = img.Height;

        int w = img.Width / size;
        int h = img.Height / size;
        double scale = 1.0 / (size * size);
        Array2D ans = new(w, h);

        // filter range
        int fa = -size / 2, fb = size / 2; // these round towards 0
        if ((size & 1) == 0) // even sized filter?
            fa++; // even, shifts right (center of filter [1,2,3,4] is 2)  (todo - makes result not 90, 180, 270 degree symmetric)
        //assert(fb-fa+1 == size);

        // loop over dest
        for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
        {
            double sum = 0;
            // loop over size
            for (int y = fa; y <= fb; ++y)
            for (int x = fa; x <= fb; ++x)
            {
                // symmetric across border, the edge pixel is repeated
                int ii = Reflect(x + i * size, ow);
                int jj = Reflect(y + j * size, oh);
                sum += img[ii, jj];
            }
            ans[i, j] = sum * scale;
        }
        return ans;
    }

    
                                                                                                                   

    #endregion

} // class SSIM                                                                                                                             
// namespace Lomont                                                                                                                           

// end of file                                                                                                                                      

