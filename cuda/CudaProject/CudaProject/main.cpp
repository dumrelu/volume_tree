#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <random>

#include "box_tree_interface.hpp"
#include "cpu_box_tree.hpp"
#include "cuda_box_tree.hpp"

int main()
{
	std::vector<std::unique_ptr<ppc::box_tree::box_tree_interface>> trees;
	trees.push_back(std::make_unique<ppc::box_tree::cpu_box_tree>());
	trees.push_back(std::make_unique<ppc::box_tree::cuda_box_tree>());

	ppc::box_tree::box_value_pairs boxValuePairs;
	ppc::box_tree::bboxes bboxes;
	ppc::box_tree::values expectedValues;

	std::random_device rd;
	std::default_random_engine engine{ rd() };
	std::uniform_int_distribution<> coordinatesDistribution{};
	std::uniform_int_distribution<> sizeDistribution{ 1, 100 };

	auto generateRandomBox = [&]()
	{
		auto x = coordinatesDistribution(engine);
		auto y = coordinatesDistribution(engine);
		auto z = coordinatesDistribution(engine);

		auto dx = sizeDistribution(engine);
		auto dy = sizeDistribution(engine);
		auto dz = sizeDistribution(engine);

		return ppc::box_tree::bbox{
			{x, y, z},
			{x + dx, y + dy, z + dz}
		};
	};

	constexpr auto N = 4096;
	for (auto i = 0; i < N; ++i)
	{
		boxValuePairs.push_back({ generateRandomBox(), i });
		expectedValues.push_back(i);
		bboxes.push_back(boxValuePairs.back().first);
	}

	for (int j = 0; j < 5; ++j)
	{
		for (auto i = 0; i < N; ++i)
		{
			bboxes.push_back(bboxes[i]);
		}
	}

	for (const auto& tree : trees)
	{
		tree->insert(boxValuePairs);
	}

	//std::sort(bboxes.begin(), bboxes.end());

	for (const auto& tree : trees)
	{
		std::cout << "Finding for " << tree->get_name() << "..." << std::endl;

		const auto beginTime = std::chrono::high_resolution_clock::now();
		auto result = tree->find(bboxes);
		const auto endTime = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
		const auto numOfResults = result.size();

		std::sort(result.begin(), result.end());
		result.erase(std::unique(result.begin(), result.end()), result.end());
		const auto isValid =
			result.size() == expectedValues.size()
			&& std::is_permutation(result.cbegin(), result.cend(), expectedValues.cbegin());

		std::cout << "\t->Found: " << numOfResults << " results" << std::endl;
		std::cout << "\t->Are results valid: " << isValid << std::endl;
		std::cout << "\t->Tree size: " << tree->size() << std::endl;
		std::cout << "\t->Time: " << duration.count() << " ms" << std::endl;
	}
}