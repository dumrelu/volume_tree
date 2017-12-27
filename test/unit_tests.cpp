#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bounding_box.hpp"
#include "host_allocator.hpp"
#include "volume_tree.hpp"

#include <functional>

using bbox = ppc::bounding_box<int>;
using point = bbox::point_type;
using box_tree = ppc::volume_tree<bbox, int, ppc::Intersect, ppc::Expand, ppc::host_allocator>;

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
	int i = 0;
	for (const auto& value : btree)
	{
		++i;
		REQUIRE(value.second == i);
	}
}

TEST_CASE("Box tree insertion of 3 elements", "[box_tree]")
{
	box_tree btree;
	btree.insert({ { { 0, 0, 0 },{ 10, 10, 10 } }, 1 });
	btree.insert({ { { 5, 5, 5 },{ 15, 15, 15 } }, 2 });
	btree.insert({ { { -1, 0, 1 },{ 10, 10, 20 } }, 3 });

	REQUIRE(btree.size() == 3);
	int i = 0;
	for (const auto& value : btree)
	{
		++i;
		REQUIRE(value.second == i);
	}
}