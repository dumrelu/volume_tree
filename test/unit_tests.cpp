#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bounding_box.hpp"
#include "host_allocator.hpp"
#include "volume_tree.hpp"

#include <array>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_set>

using bbox = ppc::bounding_box<int>;
using point = bbox::point_type;
using box_tree = ppc::volume_tree<bbox, int, ppc::Intersect, ppc::Expand, std::less<bbox>, ppc::host_allocator>;

TEST_CASE("Bounding box initialization", "[bbox]")
{
	point min{ 0, 0, 0 };
	point max{ 10, 10, 10 };
	point center{ 5, 5, 5 };
	bbox box{ min, max };

	REQUIRE(box.min == min);
	REQUIRE(box.max == max);
}

TEST_CASE("Expand a bounding box with a point", "[expand]")
{
	bbox box{ { 0, 0, 0 },{ 10, 10, 10 } };

	point toContain{ -1, 15, 5 };
	ppc::Expand{}(box, toContain);

	REQUIRE(box.min == point{ -1, 0, 0 });
	REQUIRE(box.max == point{ 10, 15, 10 });
}

TEST_CASE("Expand a bounding box with another bounding box", "[expand]")
{
	bbox box1{ { 0, 0, 0 },{ 10, 10, 10 } };
	bbox box2{ { 5, -1, 5 },{ 15, 9, 12 } };

	ppc::Expand{}(box1, box2);

	REQUIRE(box1.min == point{ 0, -1, 0 });
	REQUIRE(box1.max == point{ 15, 10, 12 });
}

TEST_CASE("Intersection between two bounding boxes", "[intersection]")
{
	bbox box{ { 0, 0, 0 },{ 10, 10, 10 } };
	bbox box1{ { 5, 5, 5 },{ 12, 13, 14 } };
	bbox box2{ { 2, 2, 2 },{ 7, 7, 7 } };
	bbox box3{ { 10, 10, 10 },{ 11, 12, 13 } };
	bbox box4{ { -5, -6, -7 },{ -1, -1, -1 } };

	REQUIRE(ppc::Intersect{}(box, box1));
	REQUIRE(ppc::Intersect{}(box, box2));
	REQUIRE(ppc::Intersect{}(box, box3));
	REQUIRE(!ppc::Intersect{}(box, box4));
}

TEST_CASE("Box tree iterator", "[box_tree]")
{
	/*
	c
	/ \
	/     \
	b0     b1
	/   \    |
	a0    a1   a2
	/  \  /  \  |
	0  1  2  3  4
	*/
	using iterator = box_tree::iterator;
	using node_type = box_tree::node_type;
	using value_type = box_tree::value_type;

	auto allocateValue = [](int i)
	{
		bbox box{ { i * 100, i * 100, i * 100 },{ (i + 1) * 100, (i + 1) * 100, (i + 1) * 100 } };
		return new value_type{ box, i };
	};

	auto setChildren = [](node_type& parent, node_type* left, node_type* right)
	{
		parent.left = left;
		if (left != nullptr)
		{
			left->parent = &parent;
		}

		parent.right = right;
		if (right != nullptr)
		{
			right->parent = &parent;
		}
	};

	node_type values[5];
	for (auto i = 0; i < 5; ++i)
	{
		values[i].value = allocateValue(i);
	}

	node_type a[3];
	setChildren(a[0], &values[0], &values[1]);
	setChildren(a[1], &values[2], &values[3]);
	setChildren(a[2], &values[4], nullptr);

	/*a[0].value = allocateValue(10);
	a[1].value = allocateValue(11);
	a[2].value = allocateValue(12);*/

	node_type b[2];
	setChildren(b[0], &a[0], &a[1]);
	setChildren(b[1], &a[2], nullptr);

	/*b[0].value = allocateValue(20);
	b[1].value = allocateValue(21);*/


	node_type c;
	setChildren(c, &b[0], &b[1]);

	iterator it{ &values[0] };
	iterator end{ nullptr };
	auto j = 0;
	for (; it != end; ++it, ++j)
	{
		REQUIRE(it->second == j);
	}
	REQUIRE(j == 5);

	for (auto i = 0; i < 5; ++i)
	{
		delete values[i].value;
	}
}

TEST_CASE("Box tree insertion of 2 elements", "[box_tree]")
{
	box_tree btree;
	btree.insert({ {{0, 0, 0}, {10, 10, 10}}, 1 });
	btree.insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });

	REQUIRE(btree.size() == 2);
	std::unordered_set<int> elements;
	for (const auto& value : btree)
	{
		elements.insert(value.second);
	}
	REQUIRE(elements.size() == btree.size());
}

TEST_CASE("Box tree insertion of 3 elements", "[box_tree]")
{
	box_tree btree;
	btree.insert({ { { 0, 0, 0 },{ 10, 10, 10 } }, 1 });
	btree.insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });
	btree.insert({ { { -1, 0, 1 },{ 10, 10, 20 } }, 3 });

	REQUIRE(btree.size() == 3);
	std::unordered_set<int> elements;
	for (const auto& value : btree)
	{
		elements.insert(value.second);
	}
	REQUIRE(elements.size() == btree.size());
}

TEST_CASE("Box tree insertion of 4 elements", "[box_tree]")
{
	box_tree btree;
	btree.insert({ { { 0, 0, 0 },{ 10, 10, 10 } }, 1 });
	btree.insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });
	btree.insert({ { { -1, 0, 1 },{ 10, 10, 20 } }, 3 });
	btree.insert({ { { 0, -2, 0 },{ 16, 10, 20 } }, 4 });

	REQUIRE(btree.size() == 4);
	std::unordered_set<int> elements;
	for (const auto& value : btree)
	{
		elements.insert(value.second);
	}
	REQUIRE(elements.size() == btree.size());
}

TEST_CASE("Box tree insertion of 5 elements", "[box_tree]")
{
	box_tree btree;
	btree.insert({ { { 0, 0, 0 },{ 10, 10, 10 } }, 1 });
	btree.insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });
	btree.insert({ { { -1, 0, 1 },{ 10, 10, 20 } }, 3 });
	btree.insert({ { { 0, -2, 0 },{ 16, 10, 20 } }, 4 });
	btree.insert({ { { 0, 0, -3 },{ 16, 17, 20 } }, 5 });

	REQUIRE(btree.size() == 5);
	std::unordered_set<int> elements;
	for (const auto& value : btree)
	{
		elements.insert(value.second);
	}
	REQUIRE(elements.size() == btree.size());
}

TEST_CASE("Box tree insertion of n elements", "[box_tree]")
{
	//Seems that the height of the tree is equal to std::ceil(std::log2(size)) + 2
	constexpr auto N = 50;

	box_tree btree;
	for (auto i = 0; i < N; ++i)
	{
		btree.insert({ { { i, i, i },{ i + 1, i + 1, i + 1 } }, i });
	}
	

	REQUIRE(btree.size() == N);
	std::unordered_set<int> elements;
	for (const auto& value : btree)
	{
		elements.insert(value.second);
	}
	REQUIRE(elements.size() == btree.size());
}

#include <iostream>

TEST_CASE("Box tree ordering", "[box_tree]")
{
	constexpr auto N = 100;
	std::array<int, N> values;
	std::iota(values.begin(), values.end(), 0);

	std::random_device rd;
	std::default_random_engine engine;
	//std::shuffle(values.begin(), values.end(), engine);

	box_tree btree;
	for (auto i : values)
	{
		btree.insert({ { { i, 0, 0 },{ i, 0, 0 } }, i });
	}
	REQUIRE(btree.size() == N);

	const auto isSorted = std::is_sorted(
		btree.cbegin(), btree.cend(), 
		[](const auto& lhs, const auto& rhs)
		{
			return lhs.second < rhs.second;
		}
	);
	REQUIRE(isSorted);
}

TEST_CASE("Finding a box", "[box_tree]")
{
	constexpr auto N = 50;

	box_tree btree;
	for (auto i = 0; i < N; ++i)
	{
		btree.insert({ { { i, i, i },{ i + 1, i + 1, i + 1 } }, i });
	}

	auto allFound = true;
	for (const auto& value : btree)
	{
		allFound = (btree.find(value.first) != btree.end());
		if (!allFound)
		{
			break;
		}
	}
	REQUIRE(allFound);

	constexpr auto center = N / 2;
	bbox similarButDifferent{ {center, center, center}, {center + 1, center + 1, center + 2} };
	REQUIRE(static_cast<const box_tree&>(btree).find(similarButDifferent) == btree.cend());

	constexpr auto outsideN = N * 2;
	bbox notIntersectingRoot{ { outsideN, outsideN, outsideN },{ outsideN + 1, outsideN + 1, outsideN + 1 } };
	REQUIRE(static_cast<const box_tree&>(btree).find(notIntersectingRoot) == btree.cend());
}