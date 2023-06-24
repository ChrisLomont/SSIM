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

/*
  Structural Similarity Index image quality metric (SSIM)
  See http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
  Also computes PSNR, MSE, RMSE image quality metrics
  C++ and C# implementations Copyright Chris Lomont 2009-2023
  based on my old C# code at https://lomont.org/software/misc/ssim/SSIM.html
  code lives at https://github.com/ChrisLomont/SSIM
  blog post at https://lomont.org/posts/2023/ssim/
  Send fixes and comments to WWW.LOMONT.ORG
*/

/* Notes:
 todo - sync
not designed for performance - designed to be correct and portable - compare your high erpformance ones to this to ensure quality matches
Notes:
- these operate on single channel grayscale
- often called Y-PSNR, Y-SSIM, etc.
- to do color: explain....
- See SSIM notes below for a lot of issues with SSIM in general
- todo - my blog post
- example code, say loading stbi headers, etc..

/* Usage
 * # https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images

 # images and test values from https://www.cns.nyu.edu/~lcv/ssim/


 # refimgs\studentsculpture.bmp jp2k\img2.bmp 0.981891

 *
 * mostly wrong
 1. Most of these metrics operate on a grayscale image
 2. Grayscale should be in linear space, except possibly for SSIM, which did not define a color space in the paper. Use gamma 2.2 there sees ok
 3. These are sometimes called Y-MSE, Y-PSNR, Y-SSIM, for the Y (grayscale) channel
 4. Simple helpers are provides for sRGB gamma to linear and linear rgb to grayscale (rec601)
 TODO
 1. explain how to do color, how most are Y-SSIM, Y-MSE, etc. avg colors?
 2. include self tests
 3. simple grayscale, gamma (sRGB, 2.2?)
 4. explain image formats, what color space they generally are in
 5. incporporate test PNG https://upload.wikimedia.org/wikipedia/commons/c/c9/Srgbnonlinearity.png, save as PPM?
 6. Simple usage code example
 */


/* SSIM Notes

Test images at http://www.cns.nyu.edu/~lcv/ssim/#test
also has (no longer linked) full database of images and SSIM scores

SSIM:
     x = {x1,x2,...,xN}, y = {y1,y2,...,yN} discrete, non-negative signals (images, etc.)
     ux, uy = mean of x and y, = (1/N) Sum xi
     σx^2, σy^2 = variance = (1/(N-1)) Sum(xi-ux)^2
     σxy = covariance = (1/(N-1))Sum(xi-ux))(yi-uy)
     then luminance, contrast, structure comparison measures:

     l(x,y) = (2 ux uy + C1)/(ux^2 + uy^2 + C1)
     c(x,y) = (2 σx σy + C2)/(σx^2 + σy^2 + C2)
     s(x,y) = (σxy + C3) / (σx σy + C3)

     C1 = (K1 L)^2, C2 = (K2 L)^2, C3 = C2/2

     L = dynamic range of values (255 for 8 bit pixels, 1 for 0-1 floating point, ...)

     K1, K2 << 1 (todo - what is std def?)

     SSIM(x,y) = l(x,y)^α * c(x,y)^β * s(x,y)^γ

     α = β = γ = 1 sets all three components equally important

     1. SSIM(x,y) = SSIM(y,x)
     2. SSIM(x,y) <= 1
     3. SSIM(x,y) = 1 iff x = y

     for MSSIM (mean SSIM)
     Compute over all M of the BxB subimages in image, return average of these. Paper (3) used B = 8

     paper (1) uses C1 = C2 = 0, but can div by 0, unstable, so (3) sets K1 = 0.01 and K2 = 0.03
     todo - get paper refs

     (3) added 11x11 gaussian weighting function wi, std dev 1.5 samples, normalized to sum(wi) = 1
     then use
     mean = Sum wi * xi
      std dev = sqrt(variance) = sqrt( Sum wi (xi-ux)^2 )
     σxy = covariance = Sum wi * (xi-ux))(yi-uy)

     (1) Z. Wang and A. C. Bovik, “A universal image quality index,” IEEE Signal Processing Letters, vol. 9, pp. 81–84, Mar. 2002.
     (2) "MULTI-SCALE STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT", Wang, Simoncelli, Bovik, 2003,
         https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
     (3) Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error measurement to structural similarity,”
         IEEE Trans. Image Processing, vol. 13, Jan. 2004. https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
     (4) "Understanding SSIM," Jim Nillson and Tomas Akenine-Moller, 2020 https://arxiv.org/pdf/2006.13846.pdf
     (5) "Mean Squared Error: Love It or Leave It?," Wang, Bovik, 2009, https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf

       NOTE from (4), "The input color space of SSIM is never defined. As the
       reference Matlab script performs no color space transformations on
       inputs, our assumption throughout this paper is that all images
       are encoded in sRGB color space, i.e., approximately gamma
       encoded with an exponent ≈ 2.4. Note that this means that an
       image that is loaded by the SSIM script is assumed to be viewed
       directly on screen as is. For two images A and B, the original
       formula [19] for per-pixel SSIM is given by


       matlab rgb2gray 0.2989 * R + 0.5870 * G + 0.1140 * B https://www.mathworks.com/help/matlab/ref/rgb2gray.html
       Rec.ITU-R BT.601-7 calculates E'y using the following formula:
       0.299 * R + 0.587 * G + 0.114 * B

     

TODO:
- add MS-SSIM, CW-SSIM
c++ reuse gaussian filter?
            // 3 channel PSNR idea https://dsp.stackexchange.com/questions/71845/defining-the-snr-or-psnr-for-color-images-3-channel-rgb-files
            // see also https://groups.google.com/g/sci.image.processing/c/0iypIGoJf7g



 */

/* History:
 * June 2023
 * 0.93 - Fix to ensure matches original on common datasets 
 *
 * Jan 2019
 * 0.92 - Made C++ version in parallel
 *      - Removed dependency on non dotnet CORE things
 *      - Made API static to make easier to use
 *      - added MSE, RMSE, PSNR metrics for ease of use
 *      - renamed to ImageMetrics.cs/.h
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
    /// get grayscale pixel in 0-1 from image index i,j
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

    // Peak Signal-to-Noise Ratio
    public static double Psnr(int width, int height, GetPixel getPixel1, GetPixel getPixel2)
    {
        // for MSE = 0, gives inf (or nan?)
        return 10.0 * Math.Log10(1.0 / Mse(width, height, getPixel1, getPixel2));
    }


    // compute SSIM from a single channel of pixels in [0,1]
    // see notes for color space, gamma, rgb to gray conversions, etc.
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
    // of the coefficient 0.299. Use this for RGB (in [0,1] per channel) to grayscale 
    // to match original algorithm
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

    // compute SSIM on one channel
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

        // automatic downsampling
        int f = (int)Math.Max(1, Math.Round(Math.Min(width, height) / 256.0));
        if (f > 1)
        {
            // simple low-pass filter, subsamples by f
            img1 = SubSample(img1, f);
            img2 = SubSample(img2, f);
        }

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


    /// <summary>
    /// Hold a 2D array of doubles, provide relevant operations
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
    /// Create a normalized Gaussian window of the given size and 
    /// standard deviation. Size must be odd
    /// </summary>
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

