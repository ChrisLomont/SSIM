// code to illustrate use of Chris Lomont's SSIM

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format> // C++20
#include "ImageMetrics.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // read images

using namespace std;
using namespace Lomont::Graphics;


class Image : vector<uint8_t>
{
public:
	int w, h, channels;
	Image(const string& filename)
	{   // read image using Sean Barret image loader
		int x, y, storedChannels;

		unsigned char* data = stbi_load(filename.c_str(), &x, &y, &storedChannels, 0); 
		channels = storedChannels;
		if (data != nullptr)
		{
			w = x; h = y;
			this->resize(w * h * channels);
			memcpy(this->data(), data, w * h * channels);
			stbi_image_free(data);
		}
		else
		{
			cerr << "Cannot find file " << filename << endl;
			throw exception();
		}
	}

	// in 0-1
	double GetGrayscale(int i, int j) const
	{
		if (channels == 1)
		{
			return (*this)[i + j * w] / 255.0;
		}
		else if (channels == 3 || channels == 4)
		{
			int index = (i + j * w) * channels;
			double r = (*this)[index] / 255.0;
			double g = (*this)[index+1] / 255.0;
			double b = (*this)[index+2] / 255.0;
			return Lomont::Graphics::ImageMetrics::Rgb2Gray(r, g, b);
		}
		return 0;
	}	
};

double SSIM(const Image& src, const Image& dst)
{
	return Lomont::Graphics::ImageMetrics::SSIM(
		src.w, src.h, // image size
		// functor to get 0-1 valued grayscale
		[&](int i, int j) {return src.GetGrayscale(i, j); },
		// functor to get 0-1 valued grayscale
		[&](int i, int j) {return dst.GetGrayscale(i, j); }
	);
}


void TestSSIM(const string & datasetPath)
{

	// Compare 6 SSIM values for the Einstein images at http://www.cns.nyu.edu/~lcv/ssim/#test

	const string filenames[] = { "einstein","meanshift","contrast","impulse","blur","jpg" };

	// correct SSIM answers from TODO
	const double correct[] = { 1.0, 0.988, 0.913, 0.840, 0.694, 0.662};

	Image src(datasetPath + "einstein.gif");
	int i = 0;
	for (auto& f : filenames)
	{
		auto filename = datasetPath + f + ".gif";
		Image dst(filename);
		
		double ssim_score = SSIM(src, dst);

		// scale back to 256 since that seems to be the testing numbers?
		auto mse_score = 255 * 255 *
			Lomont::Graphics::ImageMetrics::MSE(
				src.w, src.h, // image size
				// functor to get 0-1 valued grayscale
				[&](int i, int j) {return src.GetGrayscale(i, j) / 255.0; },
				// functor to get 0-1 valued grayscale
				[&](int i, int j) {return dst.GetGrayscale(i, j) / 255.0; }
		);

		cout << format("{} mse {:0.0f} ssim {:0.3f} (truth {:0.3f})", f, round(mse_score), ssim_score, correct[i]) << endl;
		++i;
	}
}

// the most precise way to parse strings, as of C++17
double strToDouble(const string& st)
{
	double x;
	auto [p, ec] = std::from_chars(st.data(), st.data() + st.size(), x);
	if (p == st.data()) {
		throw exception();
	}
	return x;
}

void TestSSIM_LIVE(const string& datasetPath)
{
	// Test SSIM from http://www.cns.nyu.edu/~lcv/ssim/#test
	// on the large 982 image LIVE database

	double maxerr = 0, totalerr = 0;

	// loaded, no gamma applied
	auto fname = datasetPath + "allcompares.txt"; // made by me from the LIVE  info

	std::ifstream file(fname);
	std::string str;
	int count = 0;
	while (std::getline(file, str))
	{
		// each line is two filenames and an expected score
		// C++ is ugly verbose !
		istringstream iss(str);
		string s;
		vector<string> split;
		while (getline(iss, s, ' '))
		{
			split.push_back(s);
		}
		assert(split.size() == 3);
		auto fn1 = datasetPath + split[0];
		auto fn2 = datasetPath + split[1];
		auto score = strToDouble(split[2]);

		// SSIM score them
		Image img1(fn1);
		Image img2(fn2);

		auto ssim_score = SSIM(img1, img2);
		double err = abs(score - ssim_score);
		maxerr = max(err, maxerr);
		totalerr += err;
		++count;
		cout << format("{} {} SSIM {}: correct {:0.6f} obtained {:0.6f} (max {:0.5f} avg {:0.5f})", split[0], split[1], count, score, ssim_score, maxerr, totalerr/count)  << endl;
	}
	cout << format("Done max err {} avg err {} over {} items", maxerr, totalerr / count, count) << endl;
}

int main()
{
	TestSSIM("C:\\ImageFilterTools\\datasets\\SSIM\\");
	TestSSIM_LIVE("C:\\ImageFilterTools\\datasets\\LIVE\\");
	return 0;
}