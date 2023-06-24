using System.Diagnostics;

TestSsim(@"C:\ImageFilterTools\datasets\SSIM\");
TestSsimLive(@"C:\ImageFilterTools\datasets\LIVE\");

void TestSsim(string datasetPath)
{
    string baseName = datasetPath + "einstein.gif";
    using Image<L8> src = Image.Load<L8>(baseName);

    string[] filenames = { "einstein", "meanshift", "contrast", "impulse", "blur", "jpg" };

// correct SSIM answers from TODO
    double[] correct = { 1.0, 0.988, 0.913, 0.840, 0.694, 0.662 };

    int idx = 0;
    foreach (var f in filenames)
    {
        var filename = datasetPath + f + ".gif";

        double ssimScore = SSIM<L8>(baseName,filename);

        using Image<L8> dst = Image.Load<L8>(filename);
        double mseScore =
            255*255*
            Lomont.Graphics.ImageMetrics.Mse(
            src.Width, src.Height,
            (i, j) => Grayscale(src, i, j),
            (i, j) => Grayscale(dst, i, j)
        );

        Console.WriteLine($"{f} mse {mseScore:F0} ssim {ssimScore:F3} (truth {correct[idx]:F3})");
        ++idx;
    }
}


void TestSsimLive(string datasetPath)
{
    // Test SSIM from http://www.cns.nyu.edu/~lcv/ssim/#test
    // on the large 982 image LIVE database

    double maxerr = 0, totalerr = 0;
    int count = 0;

    // loaded, no gamma applied
    var fname = datasetPath + "allcompares.txt"; // made by me from the LIVE  info
    foreach (var str in File.ReadAllLines(fname))
    {
        // each line is two filenames and an expected score
        var split = str.Split(' ');
        Trace.Assert(split.Length == 3);
        var fn1 = datasetPath + split[0];
        var fn2 = datasetPath + split[1];
        var score = Double.Parse(split[2]);

        double ssimScore = SSIM<Rgb24>(fn1,fn2);
        double err = Math.Abs(score - ssimScore);
        maxerr = Math.Max(err, maxerr);
        totalerr += err;
        ++count;
        Console.WriteLine($"{split[0]} {split[1]} SSIM {count}: correct {score:F6} obtained {ssimScore:F6} (max {maxerr:F5}, avg {totalerr/count:F5})");
    }

    Console.WriteLine($"Done max err {maxerr} avg err {totalerr/count} over {count} items");
}

double SSIM<TPixel>(string filename1, string filename2) where TPixel : unmanaged, IPixel<TPixel>
{
    // SSIM score them
    using Image<TPixel> img1 = Image.Load<TPixel>(filename1);
    using Image<TPixel> img2 = Image.Load<TPixel>(filename2);

    return 
        Lomont.Graphics.ImageMetrics.Ssim(
            img1.Width, img1.Height,
            (i, j) => Grayscale(img1, i, j),
            (i, j) => Grayscale(img2, i, j)
        );
}

double Grayscale<TPixel>(Image<TPixel> src, int i, int j) where TPixel : unmanaged, IPixel<TPixel>
{
    if (src is Image<L8> grayscale)
    {
        return grayscale[i, j].PackedValue/255.0;
    }
    if (src is Image<Rgb24> rgb)
    {
        var p = rgb[i, j];
        var r = p.R / 255.0;
        var g = p.G / 255.0;
        var b = p.B / 255.0;
        // use same color to rgb as original
        return Lomont.Graphics.ImageMetrics.Rgb2Gray(r, g, b);
    }

    throw new NotImplementedException($"Unsupported format {typeof(TPixel)}");
}
