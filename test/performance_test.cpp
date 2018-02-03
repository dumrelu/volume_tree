#include <iostream>
#include <chrono>

#include "bounding_box.hpp"
#include "host_allocator.hpp"
#include "volume_tree.hpp"

#include <vector>
#include <numeric>
#include <string>
#include <fstream>
#include <random>

using bbox = ppc::bounding_box<int>;
using point = bbox::point_type;
using box_tree = ppc::volume_tree<bbox, int, ppc::Intersect, ppc::Expand, std::less<bbox>, ppc::host_allocator>;

void write_bbox(std::ostream& stream, const bbox& box)
{
	stream << box.min.x << " " << box.min.y << " " << box.min.z << " "
		<< box.max.x << " " << box.max.y << " " << box.max.z;
}

bbox read_bbox(std::istream& stream)
{
	bbox box;
	stream >> box.min.x >> box.min.y >> box.min.z
		>> box.max.x >> box.max.y >> box.max.z;
	return box;
}

int main(int argc, char* argv[])
{
#ifndef PPC_NO_SORT_INSERT
	std::cout << "Sorting!" << std::endl;
#else
	std::cout << "No sorting!" << std::endl;
#endif

	if (argc != 2)
	{
		std::cerr << "Program requires 1 argument." << std::endl;
		return -1;
	}

	int numOfValues = -1;
	try { numOfValues = std::stoi(argv[1]); }
	catch (...) { numOfValues = -1; }

	std::vector<std::pair<bbox, int>> values;
	if (numOfValues == -1)
	{
		std::cout << "Reading from file " << argv[1] << std::endl;

		std::ifstream file{ argv[1] };
		while (file)
		{
			int value;
			auto box = read_bbox(file);
			file >> value;

			values.push_back({ box,value });
		}
	}
	else
	{
		values.reserve(numOfValues);

		std::random_device rd;
		std::default_random_engine engine;
		std::uniform_int_distribution<> distribution;
		//std::normal_distribution<> distribution{ numOfValues / 2.0, numOfValues / 10.0 };
		auto generateRandomBox = [&]()
		{
			bbox box;
			box.min.x = static_cast<int>(distribution(engine));
			box.min.y = static_cast<int>(distribution(engine));
			box.min.z = static_cast<int>(distribution(engine));

			box.min.y = 0;
			box.min.z = 0;

			box.max.x = box.min.x + 50;
			box.max.y = box.min.y + 30;
			box.max.z = box.min.z + 40;

			/*box.max.x = distribution(engine);
			box.max.y = distribution(engine);
			box.max.z = distribution(engine);

			if (box.min.x > box.max.x)
			{
				std::swap(box.min.x, box.max.x);
			}

			if (box.min.y > box.max.y)
			{
				std::swap(box.min.y, box.max.y);
			}

			if (box.min.z > box.max.z)
			{
				std::swap(box.min.z, box.max.z);
			}*/

			return box;
		};

		std::cout << "Generating random file with " << numOfValues << " values" << std::endl;
		std::ofstream file{ "random.txt" };
		for (auto i = 0; i < numOfValues; ++i)
		{
			auto val = std::make_pair(generateRandomBox(), i);
			values.push_back(val);
			write_bbox(file, val.first);
			file << " " << val.second << "\n";
		}
	}

	box_tree btree;

	{
		std::cout << "Inserting " << values.size() << " elements..." << std::endl;
		auto startTime = std::chrono::high_resolution_clock::now();
		for (const auto& value : values)
		{
			btree.insert(value);
		}
		auto endTime = std::chrono::high_resolution_clock::now();
		auto totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		std::cout << "Total insertion time: " << totalTimeMs.count() << " ms" << std::endl;
	}

	std::endl(std::cout);

	{
		std::cout << "Searching for all " << values.size() << " elements..." << std::endl;
		auto startTime = std::chrono::high_resolution_clock::now();
		for (const auto& value : values)
		{
			auto it = btree.find(value.first);
			if (it == btree.end())
			{
				std::cerr << "Cannot find element." << std::endl;
				return 1;
			}
		}
		auto endTime = std::chrono::high_resolution_clock::now();
		auto totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		std::cout << "Total search time: " << totalTimeMs.count() << " ms" << std::endl;
	}
}